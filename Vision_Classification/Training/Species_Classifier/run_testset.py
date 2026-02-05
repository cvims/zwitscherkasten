#!/usr/bin/env python3
"""
Evaluate trained classifier on test set and compute paper metrics.

Usage:
    python run_testset.py --model-dir runs-cls/effnet_b1-TIMESTAMP

Output:
    runs-cls/effnet_b1-TIMESTAMP/
    ├── test_results_paper.json
    ├── test_predictions.csv
    ├── test_confusion_matrix.npy
    └── precision_per_class.npy, recall_per_class.npy, f1_per_class.npy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms_v2(
    imgsz: int,
    mean: List[float],
    std: List[float],
) -> v2.Compose:
    """Validation transforms (no augmentation)."""
    return v2.Compose([
        v2.Resize(int(imgsz * 1.14)),
        v2.CenterCrop(imgsz),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict, Dict]:
    """Load model, config, and normalization from checkpoint."""
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    
    config = ckpt.get("config", {})
    model_type = config.get("model_type", "mobilenet_v3_small")
    num_classes = config.get("num_classes", 325)
    imgsz = config.get("imgsz", 224)
    
    # Infer num_classes from checkpoint weights
    if "classifier.3.weight" in ckpt["model"]:
        actual_num_classes = ckpt["model"]["classifier.3.weight"].shape[0]
    elif "classifier.1.weight" in ckpt["model"]:
        actual_num_classes = ckpt["model"]["classifier.1.weight"].shape[0]
    else:
        actual_num_classes = num_classes
    
    # Build model
    if model_type == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif model_type == "effnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_type == "effnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    
    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, actual_num_classes)
    
    model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    norm_info = ckpt.get("normalization", {})
    mean = norm_info.get("mean", [0.485, 0.456, 0.406])
    std = norm_info.get("std", [0.229, 0.224, 0.225])
    
    config_out = {
        "model_type": model_type,
        "num_classes": actual_num_classes,
        "imgsz": imgsz,
    }
    
    norm_out = {"mean": mean, "std": std}
    
    return model, config_out, norm_out


def make_test_loader(
    test_dir: Path,
    imgsz: int,
    batch_size: int,
    mean: List[float],
    std: List[float],
) -> Tuple[DataLoader, List[str]]:
    """Create test dataloader."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    transform = build_transforms_v2(imgsz, mean, std)
    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    class_names = dataset.classes
    return loader, class_names


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on test set."""
    model.eval()
    all_logits = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Inference", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    logits_array = np.concatenate(all_logits, axis=0)
    targets_array = np.concatenate(all_targets, axis=0)
    
    return logits_array, targets_array


def compute_paper_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute paper metrics: top-1, balanced_acc, macro precision/recall/f1."""
    predictions = logits.argmax(axis=1)
    
    metrics = {}
    metrics["top1_accuracy"] = float((predictions == targets).mean())
    
    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(targets, predictions))
    except Exception:
        pass
    
    if logits.shape[1] >= 5:
        top5_pred = np.argsort(logits, axis=1)[:, -5:]
        metrics["top5_accuracy"] = float(np.any(top5_pred == targets[:, None], axis=1).mean())
    
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average="macro", zero_division=0
        )
        metrics["macro_precision"] = float(precision)
        metrics["macro_recall"] = float(recall)
        metrics["macro_f1"] = float(f1)
    except Exception:
        pass
    
    return metrics


def compute_per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    """Compute and save per-class metrics."""
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        np.save(str(output_dir / "precision_per_class.npy"), precision)
        np.save(str(output_dir / "recall_per_class.npy"), recall)
        np.save(str(output_dir / "f1_per_class.npy"), f1)
        np.save(str(output_dir / "support_per_class.npy"), support)
    except Exception:
        pass


def save_predictions_csv(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Save per-sample predictions to CSV."""
    predictions = logits.argmax(axis=1)
    confidences = logits.max(axis=1)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "predicted_class", "confidence", "correct"])
        
        for true_idx, pred_idx, conf in zip(targets, predictions, confidences):
            true_name = class_names[true_idx]
            pred_name = class_names[pred_idx]
            correct = (true_idx == pred_idx)
            writer.writerow([true_name, pred_name, f"{conf:.4f}", correct])


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    """Compute and save confusion matrix."""
    try:
        cm = confusion_matrix(targets, predictions)
        np.save(str(output_dir / "test_confusion_matrix.npy"), cm)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test set")
    parser.add_argument("--model-dir", type=Path, required=True, help="Model directory")
    parser.add_argument("--test-dir", type=Path, default=None, help="Test directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    model_dir = args.model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    device = get_device(args.device if args.device != "auto" else "auto")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"📦 Loading model from {model_dir}...")
    model, config, norm_info = load_checkpoint(ckpt_path, device)
    mean = norm_info["mean"]
    std = norm_info["std"]
    imgsz = config["imgsz"]
    num_classes = config["num_classes"]
    
    # Find test directory
    if args.test_dir:
        test_dir = args.test_dir
    else:
        data_root = model_dir.parent.parent / "data" / "test"
        if data_root.is_dir():
            test_dir = data_root
        else:
            raise FileNotFoundError("Test directory not found")
    
    print(f"📂 Test directory: {test_dir}")
    
    # Create loader
    print(f"📊 Creating test loader...")
    test_loader, class_names = make_test_loader(test_dir, imgsz, args.batch_size, mean, std)
    print(f"   Batches: {len(test_loader)}")
    
    # Run inference
    print(f"🚀 Running inference...")
    logits, targets = run_inference(model, test_loader, device)
    print(f"   Logits: {logits.shape}, Targets: {targets.shape}")
    
    # Compute metrics
    print(f"📈 Computing metrics...")
    metrics = compute_paper_metrics(logits, targets)
    
    predictions = logits.argmax(axis=1)
    compute_per_class_metrics(targets, predictions, model_dir)
    
    # Save results
    report = {
        "test_metrics": metrics,
        "num_test_samples": int(len(targets)),
        "num_classes": num_classes,
    }
    
    with open(model_dir / "test_results_paper.json", "w") as f:
        json.dump(report, f, indent=2)
    
    save_predictions_csv(logits, targets, class_names, model_dir / "test_predictions.csv")
    save_confusion_matrix(targets, predictions, model_dir)
    
    # Print results
    print(f"\n✅ Test Results:")
    print(f"   Top-1 Accuracy: {metrics.get('top1_accuracy', 0):.4f}")
    if "top5_accuracy" in metrics:
        print(f"   Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"   Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"   Macro F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"\n📁 Results saved to {model_dir}")


if __name__ == "__main__":
    main()
