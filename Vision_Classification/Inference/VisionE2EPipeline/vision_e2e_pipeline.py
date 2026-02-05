#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

import numpy as np
import onnxruntime as ort
import cv2


def letterbox(im: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
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


def make_session(path: str, threads: int = 4, providers: Optional[List[str]] = None) -> ort.InferenceSession:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX file not found: {p.resolve()}")
    if p.suffix.lower() != ".onnx":
        raise ValueError(f"Not an .onnx file: {p.resolve()}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(p), sess_options=so, providers=providers)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (4,) xyxy
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    """
    boxes: (N,4) xyxy
    scores: (N,)
    returns indices kept
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    idxs = np.argsort(-scores)
    keep = []

    while idxs.size > 0 and len(keep) < max_det:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]

        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        idxs = rest[ious <= iou_thres]

    return np.array(keep, dtype=np.int64)


class OneStageYOLOONNX:
    """
    Single-stage multiclass YOLO pipeline.
    Keeps same output structure as your previous TwoStage pipeline:
      {
        "detections": [
          {"box_xyxy":[...], "det_conf":..., "species_id":..., "species_name":..., "cls_prob":..., "fused_score":...},
          ...
        ],
        "winner": {"policy":..., "species_name":..., "score"/"count":...} or None
      }
    """

    def __init__(
        self,
        det_onnx: str,
        classes_txt: str,
        config_json: str,
        threads: int = 4,
        providers: Optional[List[str]] = None,
    ):
        self.det = make_session(det_onnx, threads=threads, providers=providers)
        self.det_in = self.det.get_inputs()[0].name

        self.class_names = [l.strip() for l in Path(classes_txt).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.cfg = json.loads(Path(config_json).read_text(encoding="utf-8"))

        self.imgsz = int(self.cfg.get("det_imgsz", 640))
        self.det_min_conf = float(self.cfg.get("det_min_conf", 0.25))
        self.max_det = int(self.cfg.get("max_det", 15))

        # Für Raw-Outputs (5+C) brauchen wir NMS:
        self.nms_iou = float(self.cfg.get("nms_iou", 0.45))

        # Für "unknown" Handling & Winner-Scoring behalten wir diese Keys:
        self.cls_min_prob = float(self.cfg.get("cls_min_prob", 0.50))
        self.beta = float(self.cfg.get("beta", 1.0))
        self.winner_policy = str(self.cfg.get("winner_policy", "sum"))

    def _parse_dets(self, det_out0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          boxes_xyxy (N,4) in letterbox coords (imgsz space),
          scores (N,),
          cls_ids (N,)
        Supports:
          - (N,6) / (1,N,6): [x1,y1,x2,y2,score,class]
          - (N,5+C) / (1,N,5+C): [cx,cy,w,h,obj, class_probs...]
        """
        dets = np.asarray(det_out0)
        if dets.ndim == 3:
            dets = dets[0]  # (N,dim)

        if dets.size == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        dim = dets.shape[1]

        # Case A: NMS-wrapped: [x1,y1,x2,y2,score,class]
        if dim == 6:
            boxes = dets[:, 0:4].astype(np.float32)
            scores = dets[:, 4].astype(np.float32)
            cls_ids = dets[:, 5].astype(np.int32)
            return boxes, scores, cls_ids

        # Case B: Raw: [cx,cy,w,h,obj, class_probs...]
        if dim >= 6:
            cxcywh = dets[:, 0:4].astype(np.float32)
            obj = dets[:, 4].astype(np.float32)
            cls_probs = dets[:, 5:].astype(np.float32)

            cls_ids = cls_probs.argmax(axis=1).astype(np.int32)
            cls_p = cls_probs.max(axis=1).astype(np.float32)
            scores = obj * cls_p

            # cx,cy,w,h -> x1,y1,x2,y2
            cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            # filter by conf
            m = scores >= self.det_min_conf
            boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]
            if boxes.shape[0] == 0:
                return (np.zeros((0, 4), np.float32),
                        np.zeros((0,), np.float32),
                        np.zeros((0,), np.int32))

            # NMS
            keep = nms_numpy(boxes, scores, iou_thres=self.nms_iou, max_det=self.max_det)
            return boxes[keep], scores[keep], cls_ids[keep]

        raise ValueError(f"Unsupported YOLO output shape: {dets.shape}")

    def predict(self, img_rgb: np.ndarray, winner_policy: Optional[str] = None) -> Dict[str, Any]:
        if winner_policy is None:
            winner_policy = self.winner_policy

        H, W = img_rgb.shape[:2]

        # preprocess
        lb, r, left, top = letterbox(img_rgb, new_shape=self.imgsz)
        x = lb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1,3,S,S

        # infer
        det_out = self.det.run(None, {self.det_in: x})
        boxes_lb, scores, cls_ids = self._parse_dets(det_out[0])

        if boxes_lb.shape[0] == 0:
            return {"detections": [], "winner": None}

        # Map boxes back to original image space + build output
        det_list: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2), sc, ci in zip(boxes_lb, scores, cls_ids):
            # back-map from letterbox coords -> original
            x1o = (float(x1) - left) / r
            y1o = (float(y1) - top) / r
            x2o = (float(x2) - left) / r
            y2o = (float(y2) - top) / r

            x1o = float(np.clip(x1o, 0, W))
            y1o = float(np.clip(y1o, 0, H))
            x2o = float(np.clip(x2o, 0, W))
            y2o = float(np.clip(y2o, 0, H))

            box = [int(x1o), int(y1o), int(x2o), int(y2o)]

            sc = float(sc)
            # "unknown" analog wie vorher: wenn Prob zu klein
            if sc < self.cls_min_prob:
                name = "unknown"
            else:
                name = self.class_names[int(ci)] if int(ci) < len(self.class_names) else str(int(ci))

            # Behalte Felder bei:
            det_conf = sc
            cls_prob = sc  # bei One-Stage ist die "Klassenwahrscheinlichkeit" bereits im Score enthalten (typisch obj*cls)
            fused = float(det_conf) * (float(cls_prob) ** float(self.beta))  # beta=1.0 => fused == score

            det_list.append({
                "box_xyxy": box,
                "det_conf": det_conf,
                "species_id": int(ci),
                "species_name": name,
                "cls_prob": cls_prob,
                "fused_score": fused,
            })

        winner = self._pick_winner(det_list, policy=winner_policy)
        return {"detections": det_list, "winner": winner}

    @staticmethod
    def _pick_winner(dets: List[Dict[str, Any]], policy: str = "sum"):
        dets2 = [d for d in dets if d["species_name"] != "unknown"]
        if not dets2:
            return None

        if policy == "max":
            best = max(dets2, key=lambda d: d["fused_score"])
            return {"policy": "max", "species_name": best["species_name"], "score": best["fused_score"]}

        if policy == "count":
            c = {}
            for d in dets2:
                c[d["species_name"]] = c.get(d["species_name"], 0) + 1
            k = max(c.items(), key=lambda kv: kv[1])[0]
            return {"policy": "count", "species_name": k, "count": c[k]}

        # default: sum
        s = {}
        for d in dets2:
            s[d["species_name"]] = s.get(d["species_name"], 0.0) + d["fused_score"]
        k, v = max(s.items(), key=lambda kv: kv[1])
        return {"policy": "sum", "species_name": k, "score": v}


def draw_result(img_rgb: np.ndarray, out: Dict[str, Any], show_winner: bool = True) -> np.ndarray:
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


def run_video(pipe: OneStageYOLOONNX, source=0, display=True, save_path=None, skip=1, width=None, height=None):
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

        n += 1
        dt = time.time() - t0
        if dt >= 1.0:
            last_fps = n / dt
            n = 0
            t0 = time.time()
        cv2.putText(annot_bgr, f"FPS: {last_fps:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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
    BASE = Path(__file__).resolve().parent

    pipe = OneStageYOLOONNX(
        det_onnx=str(BASE / "Yolo26s.onnx"),
        classes_txt=str(BASE / "classes.txt"),
        config_json=str(BASE / "vision_pipeline_config.json"),
        threads=4,
        providers=["CPUExecutionProvider"],
    )

    # Testbild
    # img = cv2.cvtColor(cv2.imread(str(BASE / "test.jpg")), cv2.COLOR_BGR2RGB)
    # print(pipe.predict(img))

    # Live/Webcam oder Video
    run_video(pipe, source=0, display=True, width=1280, height=720, skip=1)
