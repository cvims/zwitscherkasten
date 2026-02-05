#!/usr/bin/env python3
"""
Train bird species classifiers with best hyperparameters.

Reproduces best training runs for EfficientNetB0, EfficientNetB1, MobileNetV3-Small
using fixed best hyperparameters. Computes and saves metrics without MLflow.

Usage:
    python train_species_classifier.py --model effnet_b0|effnet_b1|mobilenet_v3_small
    
Output:
    runs-cls/{model}-{timestamp}/
    ├── best.pt
    ├── last.pt
    ├── metrics.json
    ├── config.json
    ├── classes.txt
    └── normalization.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
from tqdm.auto import tqdm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


BEST_PARAMS = {
    "mobilenet_v3_small": {
        "lr": 0.0006, "batch": 32, "weight_decay": 0.0001,
        "cutmix_prob": 0.3, "cutmix_alpha": 1.0,
        "randaugment_magnitude": 6, "randaugment_ops": 2,
        "randomerasing_p": 0.25,
    },
    "effnet_b0": {
        "lr": 0.0005, "batch": 32, "weight_decay": 0.0001,
        "cutmix_prob": 0.3, "cutmix_alpha": 1.0,
        "randaugment_magnitude": 9, "randaugment_ops": 2,
        "randomerasing_p": 0.25,
    },
    "effnet_b1": {
        "lr": 0.0006, "batch": 32, "weight_decay": 0.0001,
        "cutmix_prob": 0.3, "cutmix_alpha": 1.0,
        "randaugment_magnitude": 6, "randaugment_ops": 2,
        "randomerasing_p": 0.25,
    },
}


@dataclass
class CFG:
    data_root: str = "data"
    runs_dir: str = "runs-cls"
    model_type: str = "effnet_b1"
    epochs: int = 40
    imgsz: int = 224
    workers: int = 10
    
    lr: float = 6e-4
    weight_decay: float = 3e-3
    label_smoothing: float = 0.15
    dropout_p: float = 0.4
    
    warmup_epochs: int = 2
    use_amp: bool = True
    seed: int = 42
    log_top5: bool = True
    early_stop_patience: int = 5
    
    device: str = "auto"
    norm_compute_batch: int = 64


cfg = CFG()


# =====================================================================
# METRICS
# =====================================================================
def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Compute classification metrics (top1, balanced_acc, precision, recall, f1)."""
    predictions = logits.argmax(axis=1)
    top1_acc = (predictions == targets).mean()
    
    metrics = {"top1_accuracy": float(top1_acc)}
    
    try:
        bal_acc = balanced_accuracy_score(targets, predictions)
        metrics["balanced_accuracy"] = float(bal_acc)
    except Exception:
        pass
    
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="macro", zero_division=0
        )
        metrics["macro_precision"] = float(precision)
        metrics["macro_recall"] = float(recall)
        metrics["macro_f1"] = float(f1)
    except Exception:
        pass
    
    if logits.shape[1] >= 5:
        top5_pred = np.argsort(logits, axis=1)[:, -5:]
        top5_acc = np.any(top5_pred == targets[:, None], axis=1).mean()
        metrics["top5_accuracy"] = float(top5_acc)
    
    return metrics


def compute_per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: Optional[Path] = None,
    epoch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class precision, recall, f1 and support."""
    try:
        p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        
        try:
            np.save(output_dir / f"precision_per_class{epoch_str}.npy", p_per_class)
            np.save(output_dir / f"recall_per_class{epoch_str}.npy", r_per_class)
            np.save(output_dir / f"f1_per_class{epoch_str}.npy", f1_per_class)
            np.save(output_dir / f"support{epoch_str}.npy", support)
        except Exception:
            pass
    
    return p_per_class, r_per_class, f1_per_class, support


# =====================================================================
# TRANSFORMS
# =====================================================================
def build_transforms_v2(
    imgsz: int,
    augment: bool,
    mean: List[float],
    std: List[float],
    randaugment_magnitude: int = 9,
    randaugment_ops: int = 2,
    randomerasing_p: float = 0.25,
) -> Tuple[v2.Compose, v2.Compose]:
    """Build v2 transforms: tensor-first, no numpy."""
    
    train_tfms = [
        v2.RandomResizedCrop(imgsz, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        v2.RandomPerspective(distortion_scale=0.15, p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.RandomErasing(p=randomerasing_p, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ]

    val_tfms = [
        v2.Resize(int(imgsz * 1.14)),
        v2.CenterCrop(imgsz),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]

    return v2.Compose(train_tfms), v2.Compose(val_tfms)


# =====================================================================
# UTILS
# =====================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}") if torch.cuda.is_available() else torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...]) -> Dict[int, float]:
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res: Dict[int, float] = {}
    for k in topk:
        k = min(k, logits.size(1))
        res[k] = correct[:k].reshape(-1).float().sum().item() / targets.numel()
    return res


def cosine_warmup_lr(epoch: int, step: int, steps_per_epoch: int, base_lr: float, warmup_epochs: int) -> float:
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    cur = epoch * steps_per_epoch + step

    if warmup_steps > 0 and cur < warmup_steps:
        return base_lr * (cur + 1) / warmup_steps

    t = (cur - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val: float,
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_top1": best_val,
            "config": {
                "model_type": cfg.model_type,
                "num_classes": 253,
                "imgsz": cfg.imgsz,
            },
            "normalization": {
                "mean": norm_mean,
                "std": norm_std,
            },
        }, str(path))
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Failed to save checkpoint {path}: {e}")


def make_loaders(
    root: Path,
    mean: List[float],
    std: List[float],
    batch_size: int,
    randaugment_magnitude: int = 9,
    randaugment_ops: int = 2,
    randomerasing_p: float = 0.25,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Train/Val directories not found in {root}")

    train_t, val_t = build_transforms_v2(
        cfg.imgsz, True, mean=mean, std=std,
        randaugment_magnitude=randaugment_magnitude,
        randaugment_ops=randaugment_ops,
        randomerasing_p=randomerasing_p,
    )
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_t)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_t)

    pin = torch.cuda.is_available()
    prefetch = 2 if batch_size >= 96 else (3 if batch_size >= 64 else 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin,
        prefetch_factor=prefetch,
        persistent_workers=(cfg.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin,
        prefetch_factor=prefetch,
        persistent_workers=(cfg.workers > 0),
    )

    return train_loader, val_loader, len(train_ds.classes), train_ds.classes


def build_model(num_classes: int, model_type: str) -> nn.Module:
    if model_type == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    elif model_type == "effnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    elif model_type == "effnet_b1":
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
        model = models.efficientnet_b1(weights=weights)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    dropout_p = cfg.dropout_p
    in_features = model.classifier[-1].in_features

    layers = []
    has_dropout = False
    for layer in model.classifier[:-1]:
        if isinstance(layer, nn.Dropout):
            layers.append(nn.Dropout(p=dropout_p))
            has_dropout = True
        else:
            layers.append(layer)

    if not has_dropout:
        layers.append(nn.Dropout(p=dropout_p))

    layers.append(nn.Linear(in_features, num_classes))
    model.classifier = nn.Sequential(*layers)
    return model


def autocast_settings(device: torch.device) -> tuple[bool, torch.dtype]:
    use_amp = cfg.use_amp and device.type in ("cuda", "mps")
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return use_amp, dtype


def validate_output_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        test_file = run_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Cannot write to output directory {run_dir}: {e}")


@torch.no_grad()
def compute_train_mean_std(
    train_dir: Path,
    imgsz: int,
    batch_size: int,
    workers: int = 0,
) -> Tuple[List[float], List[float]]:
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    tfm = transforms.Compose([
        transforms.Resize(int(imgsz * 1.14)),
        transforms.CenterCrop(imgsz),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    ds = datasets.ImageFolder(str(train_dir), transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    ch_sum = torch.zeros(3, dtype=torch.float64)
    ch_sumsq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0

    for images, _ in tqdm(loader, desc="Computing normalization", leave=False):
        b, _, h, w = images.shape
        images = images.to(dtype=torch.float64)
        n_pixels += b * h * w
        ch_sum += images.sum(dim=(0, 2, 3))
        ch_sumsq += (images * images).sum(dim=(0, 2, 3))

    mean = (ch_sum / max(1, n_pixels)).tolist()
    var = (ch_sumsq / max(1, n_pixels) - torch.tensor(mean, dtype=torch.float64) ** 2).clamp_min(1e-12)
    std = torch.sqrt(var).tolist()
    return [float(m) for m in mean], [float(s) for s in std]


# =====================================================================
# CutMix
# =====================================================================
def _rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def maybe_apply_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    cutmix_prob: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation."""
    if cutmix_alpha <= 0 or random.random() > cutmix_prob:
        return images, targets, targets, 1.0

    bs = images.size(0)
    if bs < 2:
        return images, targets, targets, 1.0

    idx = torch.randperm(bs, device=images.device)
    y_b = targets[idx]

    lam = random.betavariate(cutmix_alpha, cutmix_alpha)
    _, _, H, W = images.shape
    x1, y1, x2, y2 = _rand_bbox(W, H, lam)

    images_mixed = images.clone()
    images_mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - area / float(W * H)
    lam = float(max(0.0, min(1.0, lam)))
    return images_mixed, targets, y_b, lam


# =====================================================================
# TRAINING
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    base_lr: float,
    cutmix_prob: float,
    cutmix_alpha: float,
) -> Dict[str, float]:
    model.train()
    loss_sum = top1_sum = top5_sum = 0.0
    n = 0
    steps_per_epoch = len(loader)

    use_amp, amp_dtype = autocast_settings(device)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False, dynamic_ncols=True)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr_now = cosine_warmup_lr(epoch, step, steps_per_epoch, base_lr, cfg.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        images_cm, y_a, y_b, lam = maybe_apply_cutmix(images, targets, cutmix_prob, cutmix_alpha)

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(images_cm)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

        if scaler is not None and device.type == "cuda" and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        loss_sum += loss.item() * bs

        topk = (1, 5) if (cfg.log_top5 and logits.size(1) >= 5) else (1,)
        accs = accuracy_topk(logits.detach(), y_a, topk=topk)
        top1_sum += accs[1] * bs
        if 5 in accs:
            top5_sum += accs[5] * bs

        n += bs

        postfix = {"loss": f"{loss_sum/max(1,n):.4f}", "top1": f"{top1_sum/max(1,n):.4f}"}
        if 5 in accs:
            postfix["top5"] = f"{top5_sum/max(1,n):.4f}"
        pbar.set_postfix(postfix)

    out = {"train_loss": loss_sum / max(1, n), "train_top1": top1_sum / max(1, n)}
    if cfg.log_top5 and top5_sum > 0:
        out["train_top5"] = top5_sum / max(1, n)
    return out


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Validate and accumulate logits/targets for metrics computation."""
    model.eval()
    loss_sum = top1_sum = top5_sum = 0.0
    n = 0
    
    all_logits = []
    all_targets = []

    use_amp, amp_dtype = autocast_settings(device)

    pbar = tqdm(loader, desc=f"Val   {epoch+1}/{cfg.epochs}", leave=False, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, targets)

        bs = targets.size(0)
        loss_sum += loss.item() * bs
        top1_sum += (logits.argmax(1) == targets).sum().item()

        if logits.size(1) >= 5:
            top5_pred = logits.topk(5, dim=1)[1]
            top5_sum += (top5_pred == targets.view(-1, 1)).any(1).sum().item()

        n += bs
        
        # Accumulate for metrics computation
        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        postfix = {"loss": f"{loss_sum/max(1,n):.4f}", "top1": f"{top1_sum/max(1,n):.4f}"}
        if top5_sum > 0:
            postfix["top5"] = f"{top5_sum/max(1,n):.4f}"
        pbar.set_postfix(postfix)

    out = {"val_loss": loss_sum / max(1, n), "val_top1": top1_sum / max(1, n)}
    if cfg.log_top5 and top5_sum > 0:
        out["val_top5"] = top5_sum / max(1, n)
    
    # Concatenate all logits and targets
    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    targets_array = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    
    return out, logits_array, targets_array


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BEST_PARAMS.keys()), default="effnet_b1")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    cfg.model_type = args.model
    cfg.data_root = args.data_root
    cfg.device = args.device
    
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_seed(cfg.seed)
    
    device = get_device(cfg.device)
    data_root = Path(cfg.data_root)
    hp = BEST_PARAMS[cfg.model_type]
    
    cfg.lr = hp["lr"]
    cfg.weight_decay = hp["weight_decay"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.runs_dir) / f"{cfg.model_type}-{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"Training {cfg.model_type}")
    print(f"LR={hp['lr']:.6f}, Batch={hp['batch']}, WD={hp['weight_decay']:.6f}")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")
    
    print("Validating output directory...")
    validate_output_dir(run_dir)
    
    config_dict = {
        "model_type": cfg.model_type,
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "hyperparams": hp,
        "dropout_p": cfg.dropout_p,
    }
    try:
        (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Cannot save config.json: {e}")
    
    print("Computing normalization...")
    train_dir = data_root / "train"
    norm_mean, norm_std = compute_train_mean_std(train_dir, cfg.imgsz, cfg.workers, cfg.norm_compute_batch)
    try:
        (run_dir / "normalization.json").write_text(json.dumps({"mean": norm_mean, "std": norm_std}, indent=2))
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Cannot save normalization.json: {e}")
    
    train_loader, val_loader, num_classes, class_names = make_loaders(
        data_root, norm_mean, norm_std, hp["batch"],
        hp["randaugment_magnitude"], hp["randaugment_ops"], hp["randomerasing_p"],
    )
    try:
        (run_dir / "classes.txt").write_text("\n".join(class_names))
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Cannot save classes.txt: {e}")
    
    model = build_model(num_classes, cfg.model_type).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    
    best_val_top1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    final_val_logits = None
    final_val_targets = None
    
    for epoch in range(cfg.epochs):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, cfg.lr, hp["cutmix_prob"], hp["cutmix_alpha"])
        val_m, val_logits, val_targets = validate(model, val_loader, criterion, device, epoch)
        
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_val_top1, norm_mean, norm_std)
        
        if val_m["val_top1"] > best_val_top1:
            best_val_top1 = val_m["val_top1"]
            best_epoch = epoch
            bad_epochs = 0
            final_val_logits = val_logits.copy()
            final_val_targets = val_targets.copy()
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_val_top1, norm_mean, norm_std)
        else:
            bad_epochs += 1
        
        if cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Compute final metrics using accumulated logits/targets from best epoch
    metrics = {
        "best_val_top1": float(best_val_top1),
        "best_epoch": int(best_epoch),
        "total_epochs": cfg.epochs,
    }
    
    if final_val_logits is not None and final_val_targets is not None:
        comprehensive_metrics = compute_metrics(final_val_logits, final_val_targets, num_classes)
        metrics.update(comprehensive_metrics)
        
        # Save per-class metrics
        predictions = final_val_logits.argmax(axis=1)
        compute_per_class_metrics(final_val_targets, predictions, output_dir=run_dir, epoch=best_epoch)
    
    if "val_top5" in val_m:
        metrics["best_val_top5"] = float(val_m["val_top5"])
    
    try:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Cannot save metrics.json: {e}")
    
    print(f"\n{'='*70}")
    print(f"Complete: best_val_top1={best_val_top1:.4f} @ epoch {best_epoch+1}")
    print(f"Metrics saved to {run_dir / 'metrics.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
