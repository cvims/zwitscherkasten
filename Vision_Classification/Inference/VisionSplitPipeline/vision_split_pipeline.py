#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import cv2


def letterbox(im: np.ndarray, new_shape: int = 984, color=(114, 114, 114)):
    """YOLO-style letterbox. Returns resized image + scale + padding for back-mapping."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - nw
    pad_h = new_shape - nh
    left = pad_w // 2
    top = pad_h // 2

    out = cv2.copyMakeBorder(im_resized, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, left, top


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def make_session(path: str, threads: int = 4, providers: Optional[List[str]] = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


class TwoStageONNX:
    """
    Two-stage pipeline:
      Det (YOLO ONNX) -> crop(s) -> Cls (MobileNet ONNX)

    Det output expected:
      (1,N,6) or (N,6) with [x1,y1,x2,y2,score,class]
    """

    def __init__(
        self,
        det_onnx: str,
        cls_onnx: str,
        classes_txt: str,
        config_json: str,
        threads: int = 4,
        providers: Optional[List[str]] = None,
    ):
        self.det = make_session(det_onnx, threads=threads, providers=providers)
        self.cls = make_session(cls_onnx, threads=threads, providers=providers)

        self.det_in = self.det.get_inputs()[0].name
        self.cls_in = self.cls.get_inputs()[0].name

        self.class_names = [l.strip() for l in Path(classes_txt).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.cfg = json.loads(Path(config_json).read_text(encoding="utf-8"))

        # Defaults
        self.imgsz = int(self.cfg.get("det_imgsz", 984))
        self.det_min_conf = float(self.cfg.get("det_min_conf", 0.25))
        self.max_det = int(self.cfg.get("max_det", 15))
        self.pad = float(self.cfg.get("pad", 0.15))
        self.min_crop = int(self.cfg.get("min_crop", 48))
        self.cls_imgsz = int(self.cfg.get("cls_imgsz", 224))
        self.cls_min_prob = float(self.cfg.get("cls_min_prob", 0.50))
        self.beta = float(self.cfg.get("beta", 1.5))
        self.winner_policy = str(self.cfg.get("winner_policy", "sum"))

        # Optional tuning
        self.bird_class_id = int(self.cfg.get("bird_class_id", 14))      # COCO: bird
        self.only_top1_det = bool(self.cfg.get("only_top1_det", False))  # speed-up

        self.mean = np.array(self.cfg.get("cls_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std  = np.array(self.cfg.get("cls_std",  [0.229, 0.224, 0.225]), dtype=np.float32)

    def predict(self, img_rgb: np.ndarray, winner_policy: Optional[str] = None) -> Dict[str, Any]:
        if winner_policy is None:
            winner_policy = self.winner_policy

        H, W = img_rgb.shape[:2]

        # DET preprocess
        lb, r, left, top = letterbox(img_rgb, new_shape=self.imgsz)
        x = lb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1,3,S,S

        # DET infer
        det_out = self.det.run(None, {self.det_in: x})
        dets = np.asarray(det_out[0])
        if dets.ndim == 3:
            dets = dets[0]
        if dets.size == 0:
            return {"detections": [], "winner": None}

        # expected columns: x1 y1 x2 y2 score class
        dets = dets[dets[:, 4] >= self.det_min_conf]
        dets = dets[dets[:, 5].astype(np.int32) == self.bird_class_id]
        if dets.size == 0:
            return {"detections": [], "winner": None}

        dets = dets[np.argsort(-dets[:, 4])]
        dets = dets[: self.max_det]
        if self.only_top1_det:
            dets = dets[:1]

        # Crop list
        crops = []
        meta = []  # (box_xyxy_orig, det_conf)
        for x1, y1, x2, y2, score, _clsid in dets:
            # back-map to original
            x1o = (x1 - left) / r
            y1o = (y1 - top) / r
            x2o = (x2 - left) / r
            y2o = (y2 - top) / r

            x1o = float(np.clip(x1o, 0, W))
            y1o = float(np.clip(y1o, 0, H))
            x2o = float(np.clip(x2o, 0, W))
            y2o = float(np.clip(y2o, 0, H))

            bw, bh = (x2o - x1o), (y2o - y1o)
            if bw <= 1 or bh <= 1:
                continue

            # padding
            x1p = int(max(0, x1o - self.pad * bw))
            y1p = int(max(0, y1o - self.pad * bh))
            x2p = int(min(W, x2o + self.pad * bw))
            y2p = int(min(H, y2o + self.pad * bh))

            crop = img_rgb[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            if crop.shape[0] < self.min_crop or crop.shape[1] < self.min_crop:
                continue

            crop_resized = cv2.resize(crop, (self.cls_imgsz, self.cls_imgsz), interpolation=cv2.INTER_LINEAR)
            crops.append(crop_resized)
            meta.append(((x1p, y1p, x2p, y2p), float(score)))

        if not crops:
            return {"detections": [], "winner": None}

        # CLS preprocess
        b = np.stack(crops, axis=0).astype(np.float32) / 255.0  # N,H,W,3
        b = np.transpose(b, (0, 3, 1, 2))                       # N,3,H,W
        b = (b - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        # CLS infer
        logits = self.cls.run(None, {self.cls_in: b})[0]  # N,C
        probs = softmax(logits, axis=1)
        top_i = probs.argmax(axis=1)
        top_p = probs.max(axis=1)

        # Fuse + output
        det_list: List[Dict[str, Any]] = []
        for (box, det_conf), ci, cp in zip(meta, top_i, top_p):
            cp = float(cp)
            if cp < self.cls_min_prob:
                name = "unknown"
            else:
                idx = int(ci)
                name = self.class_names[idx] if idx < len(self.class_names) else str(idx)

            fused = float(det_conf) * (cp ** self.beta)
            det_list.append({
                "box_xyxy": list(map(int, box)),
                "det_conf": float(det_conf),
                "species_id": int(ci),
                "species_name": name,
                "cls_prob": cp,
                "fused_score": float(fused),
            })

        winner = self._pick_winner(det_list, policy=winner_policy)
        return {"detections": det_list, "winner": winner}

    @staticmethod
    def _pick_winner(dets: List[Dict[str, Any]], policy: str = "sum") -> Optional[Dict[str, Any]]:
        dets2 = [d for d in dets if d["species_name"] != "unknown"]
        if not dets2:
            return None

        if policy == "max":
            best = max(dets2, key=lambda d: d["fused_score"])
            return {"policy": "max", "species_name": best["species_name"], "score": best["fused_score"]}

        if policy == "count":
            c: Dict[str, int] = {}
            for d in dets2:
                c[d["species_name"]] = c.get(d["species_name"], 0) + 1
            k = max(c.items(), key=lambda kv: kv[1])[0]
            return {"policy": "count", "species_name": k, "count": c[k]}

        # default: sum
        s: Dict[str, float] = {}
        for d in dets2:
            s[d["species_name"]] = s.get(d["species_name"], 0.0) + d["fused_score"]
        k, v = max(s.items(), key=lambda kv: kv[1])
        return {"policy": "sum", "species_name": k, "score": v}


def draw_result(img_rgb: np.ndarray, out: Dict[str, Any], show_winner: bool = True) -> np.ndarray:
    """Draw boxes + labels + optional winner overlay on RGB image; returns RGB image."""
    img = img_rgb.copy()

    for d in out.get("detections", []):
        x1, y1, x2, y2 = d["box_xyxy"]
        name = d.get("species_name", "unknown")
        det_conf = float(d.get("det_conf", 0.0))
        cls_prob = float(d.get("cls_prob", 0.0))
        fused = float(d.get("fused_score", 0.0))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} det={det_conf:.2f} cls={cls_prob:.2f} fused={fused:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(img, (x1, y_text), (x1 + tw + 4, y_text + th + baseline + 4), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y_text + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if show_winner and out.get("winner") is not None:
        w = out["winner"]
        if "count" in w:
            text = f"WINNER ({w['policy']}): {w['species_name']} count={w['count']}"
        else:
            text = f"WINNER ({w['policy']}): {w['species_name']} score={float(w.get('score', 0.0)):.2f}"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (10, 10), (10 + tw + 20, 10 + th + baseline + 20), (255, 255, 255), -1)
        cv2.putText(img, text, (20, 10 + th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def run_image(pipe: TwoStageONNX, image_path: str, out_path: Optional[str] = None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    out = pipe.predict(img_rgb)
    annot_rgb = draw_result(img_rgb, out)
    annot_bgr = cv2.cvtColor(annot_rgb, cv2.COLOR_RGB2BGR)

    if out_path:
        cv2.imwrite(out_path, annot_bgr)
        print(f"Saved: {out_path}")
    else:
        cv2.imshow("result", annot_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(out)


def run_video(pipe: TwoStageONNX, source: Union[int, str] = 0, save_path: Optional[str] = None,
              display: bool = True, skip: int = 1, width: Optional[int] = None, height: Optional[int] = None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    writer = None
    t0 = time.time()
    n = 0
    last_fps = 0.0
    i = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        i += 1
        if skip > 1 and (i % skip != 0):
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = pipe.predict(frame_rgb)
        annot_rgb = draw_result(frame_rgb, out)
        annot_bgr = cv2.cvtColor(annot_rgb, cv2.COLOR_RGB2BGR)

        # FPS
        n += 1
        dt = time.time() - t0
        if dt >= 1.0:
            last_fps = n / dt
            n = 0
            t0 = time.time()
        cv2.putText(annot_bgr, f"FPS: {last_fps:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Save video
        if save_path and writer is None:
            h, w = annot_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, 25.0, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter: {save_path}")
        if writer is not None:
            writer.write(annot_bgr)

        if display:
            cv2.imshow("pipeline", annot_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ---- Set paths like in your current script ----
    pipe = TwoStageONNX(
        det_onnx="yolo11n.onnx",
        cls_onnx="effnet_b0_253cls.onnx",  # <-- prefer fp32/fp16 or QDQ on Pi
        #cls_onnx="mobilenet_v3_small_253cls.onnx",
        classes_txt="classes.txt",
        config_json="vision_pipeline_config.json",
        threads=4,
        providers=["CPUExecutionProvider"],
    )

    # 1) Single image test:
    #run_image(pipe, ">test_image>.jpg", out_path="result.jpg")

    # 2) Live test: webcam index 0 (or pass a video file path)
    run_video(pipe, source=0, display=True, save_path=None, skip=1, width=1280, height=720)
