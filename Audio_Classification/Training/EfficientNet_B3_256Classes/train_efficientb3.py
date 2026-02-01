"""
EfficientNet-B3 Bioacoustics ONNX Export MAX PERFORMANCE 256 CLASSES!
Adapted from B0 for higher accuracy on bird species spectrograms.
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from timm import create_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).parent.resolve()

# EXPERIMENT DIR: training/efficientnet_b3_anti_overfit
CONFIG = {
    "num_classes": 256,  # 256 BIRD SPECIES
    "data": {
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "preprocessed_mels_dir": "preprocessed_mels",
    },
    "training": {
        "batch_size": 16,  # to run on RTX 6000 ADA with 48GB VRAM
        "epochs": 50,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "max_time_steps": 1024,
        "early_stop_patience": 6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "label_smoothing": 0.1,
        "num_workers": 4,  # set to 0 if you suspect dataloader multiprocessing issues
        "pin_memory": True,
    },
}

TITLE = "=" * 80


def setup_experiment_dir() -> Path:
    """
    Creates an experiment directory under CURRENT_DIR/training/...
    Uses parents=True so missing parent directories are created.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = CURRENT_DIR / "training"
    exp_dir = exp_root / f"efficientnet_b3_anti_overfit_{timestamp}"

    print(f"🚀 Creating ISOLATED experiment: {exp_dir.name}")
    print("📱 MobileNet data 100% PROTECTED!")

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "onnx").mkdir(parents=True, exist_ok=True)

    existing_exps = [d.name for d in exp_root.glob("efficientnet_*") if d.is_dir()]
    print(f"📂 Existing: {len(existing_exps)} experiments")
    if len(existing_exps) < 5:
        print(f"✅ ISOLATED Experiment: {exp_dir.absolute()}")

    return exp_dir


class PerformanceMelDataset(Dataset):
    def __init__(self, csv_file, preprocessed_dir, max_time=1024, augment=False):
        self.df = pd.read_csv(csv_file)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_time = int(max_time)
        self.augment = augment

        # Minimal schema validation (fail fast with clear message)
        required_cols = {"filepath", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV {csv_file} missing columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def _fix_length(self, mel_2d: np.ndarray) -> np.ndarray:
        """
        mel_2d: [n_mels, time]
        Returns: [n_mels, max_time]
        """
        if mel_2d.shape[1] < self.max_time:
            pad_width = self.max_time - mel_2d.shape[1]
            mel_2d = np.pad(mel_2d, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel_2d = mel_2d[:, : self.max_time]
        return mel_2d

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = self.preprocessed_dir / row["filepath"]

        mel = np.load(filepath).astype(np.float32)  # expected [128, T]
        mel = self._fix_length(mel)

        # scale and add channel -> [1, 128, 1024]
        mel = torch.from_numpy(mel / 255.0).unsqueeze(0)

        if self.augment:
            # Time mask
            if torch.rand(1) < 0.5:
                t_mask = int(self.max_time * 0.15)
                t_start = torch.randint(0, max(1, self.max_time - t_mask + 1), (1,)).item()
                mel[:, :, t_start : t_start + t_mask] = 0.0

            # Freq mask
            if torch.rand(1) < 0.5:
                f_mask = 12
                f_start = torch.randint(0, max(1, 128 - f_mask + 1), (1,)).item()
                mel[:, f_start : f_start + f_mask, :] = 0.0

            # Time stretch (robust: always returns exactly max_time)
            if torch.rand(1) < 0.3:
                stretch_factor = 1.1 if torch.rand(1) < 0.5 else 0.9
                new_time = max(8, int(self.max_time * stretch_factor))

                stretched = torch.nn.functional.interpolate(
                    mel.unsqueeze(0),  # [1, 1, 128, T]
                    size=(128, new_time),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # [1, 128, new_time]

                if new_time > self.max_time:
                    # random crop back to max_time
                    start = torch.randint(0, new_time - self.max_time + 1, (1,)).item()
                    mel = stretched[:, :, start : start + self.max_time]
                elif new_time < self.max_time:
                    # pad to max_time
                    pad_right = self.max_time - new_time
                    mel = torch.nn.functional.pad(stretched, (0, pad_right))
                else:
                    mel = stretched

        return mel, torch.tensor(int(row["label"]), dtype=torch.long)


class EfficientNetB3Audio(nn.Module):
    def __init__(self, num_classes=256, pretrained=True):
        super().__init__()
        self.backbone = create_model(
            "efficientnet_b3", pretrained=pretrained, num_classes=0, in_chans=1
        )

        # timm handles in_chans, but keep explicit stem replacement if desired
        with torch.no_grad():
            self.backbone.conv_stem.conv = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.num_features, 1280),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1280, 640),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(640, num_classes),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train")
    for mels, labels in pbar:
        mels = mels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(mels)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        mem_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        pbar.set_postfix(
            {"loss": f"{loss:.3f}", "acc": f"{100 * correct / total:.1f}", "mem": f"{mem_gb:.1f}G"}
        )
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc="Valid")
    with torch.no_grad():
        for mels, labels in pbar:
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(mels)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            top5_correct += (
                logits.topk(5, 1)[1].eq(labels.unsqueeze(1)).any(1).sum().item()
            )
            total += labels.size(0)

            all_logits.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / max(1, total)
    top1_acc = 100.0 * correct / max(1, total)
    top5_acc = 100.0 * top5_correct / max(1, total)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    preds = all_logits.argmax(1)
    bal_acc = balanced_accuracy_score(all_labels, preds)

    # cMAP macro (will be 0.0 if metric cannot be computed)
    try:
        ap_scores = average_precision_score(all_labels, all_logits, average=None)
        c_map_macro = float(np.nanmean(ap_scores))
    except ValueError:
        c_map_macro = 0.0

    return avg_loss, top1_acc, top5_acc, 100.0 * bal_acc, c_map_macro


def save_checkpoint(exp_dir, model, optimizer, epoch, val_acc, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "config": CONFIG,
        "model_type": "EfficientNetB3Audio",
        "experiment_id": exp_dir.name,
        "num_classes": CONFIG["num_classes"],
    }
    filename = f"EFFICIENTNET_B3_epoch{epoch:02d}_val{val_acc:.1f}.pth"
    torch.save(checkpoint, exp_dir / "checkpoints" / filename)
    if is_best:
        torch.save(checkpoint, exp_dir / "checkpoints" / "BEST_EFFICIENTNET_B3.pth")
        torch.save(checkpoint, exp_dir / "checkpoints" / "BEST_model.pth")


def save_metrics(exp_dir, metrics_df):
    csv_path = exp_dir / "metrics" / "training_metrics.csv"
    json_path = exp_dir / "metrics" / "training_metrics.json"
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_json(json_path, orient="records", indent=2)


def log_to_file(exp_dir, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    with open(exp_dir / "logs" / "training.log", "a") as f:
        f.write(log_entry + "\n")
    print(log_entry.strip())


def export_to_onnx(exp_dir, model, device, input_shape=(1, 1, 128, 1024)):
    onnx_dir = exp_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 90)
    print("Exporting EfficientNet-B3 to ONNX...")
    print("═" * 90)

    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    onnx_path = onnx_dir / "efficientnet_b3_audio.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 3: "time_steps"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"ONNX saved: {onnx_path}")

    metadata = {
        "model_name": "EfficientNetB3Audio",
        "num_classes": CONFIG["num_classes"],
        "input_shape": list(input_shape),
        "backbone": "efficientnet_b3 (timm)",
        "total_params": sum(p.numel() for p in model.parameters()),
        "input_description": "Mel spectrogram [batch, 1ch, 128x1024]",
        "export_date": datetime.now().isoformat(),
    }

    with open(onnx_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata saved!")
    print("═" * 90)
    print("EfficientNet-B3 ONNX export COMPLETE!")
    print("═" * 90)


def main():
    print(TITLE)
    print("SAFETY CHECK - Protecting prior EfficientNet-B0...")

    # Properly detect previous B0 experiments under CURRENT_DIR/training
    b0_root = CURRENT_DIR / "training"
    b0_dirs = list(b0_root.glob("efficientnet_b0_anti_overfit*")) if b0_root.exists() else []
    if b0_dirs:
        b0_dir = sorted(b0_dirs)[-1]
        checkpoints_dir = b0_dir / "checkpoints"
        checkpoints = list(checkpoints_dir.glob("*.pth")) if checkpoints_dir.exists() else []
        print(f"B0 checkpoints: {len(checkpoints)} files")
        print(f"B0 dir: {b0_dir.absolute()}")

    print("All prior data SAFE!")
    print(TITLE)

    exp_dir = setup_experiment_dir()
    log_to_file(exp_dir, "Starting ISOLATED EfficientNet-B3 256 classes")

    device = torch.device(CONFIG["training"]["device"])

    train_csv = CURRENT_DIR / CONFIG["data"]["train_csv"]
    val_csv = CURRENT_DIR / CONFIG["data"]["val_csv"]
    pre_dir = CURRENT_DIR / CONFIG["data"]["preprocessed_mels_dir"]

    # Fail fast if paths are wrong
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")
    if not pre_dir.exists():
        raise FileNotFoundError(f"Preprocessed mels dir not found: {pre_dir}")

    train_dataset = PerformanceMelDataset(
        str(train_csv),
        str(pre_dir),
        CONFIG["training"]["max_time_steps"],
        augment=True,
    )
    val_dataset = PerformanceMelDataset(
        str(val_csv),
        str(pre_dir),
        CONFIG["training"]["max_time_steps"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=CONFIG["training"]["num_workers"],
        pin_memory=CONFIG["training"]["pin_memory"],
        persistent_workers=CONFIG["training"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        num_workers=CONFIG["training"]["num_workers"],
        pin_memory=CONFIG["training"]["pin_memory"],
        persistent_workers=CONFIG["training"]["num_workers"] > 0,
    )

    log_to_file(exp_dir, f"📊 Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")

    model = EfficientNetB3Audio(CONFIG["num_classes"], pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_to_file(exp_dir, f"🤖 EfficientNet-B3: {total_params:,} params ({trainable_params:,} trainable)")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["training"]["label_smoothing"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0.0
    patience_counter = 0
    metrics_list = []

    log_to_file(exp_dir, "EfficientNet-B3 PERFORMANCE training STARTED...")

    for epoch in range(CONFIG["training"]["epochs"]):
        print(f"\n📈 Epoch {epoch + 1}/{CONFIG['training']['epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top5, val_bal_acc, val_cmap = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)

        gap = train_acc - val_top1
        mem_used = (
            torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_bal_acc": val_bal_acc,
            "val_cmap": val_cmap,
            "train_val_gap": gap,
            "lr": optimizer.param_groups[0]["lr"],
            "memory_gb": mem_used,
        }
        metrics_list.append(metrics)

        log_msg = (
            f"Epoch {epoch + 1}: Train {train_acc:.1f}% | "
            f"Val Top-1 {val_top1:.1f}% Top-5 {val_top5:.1f}% "
            f"BalAcc {val_bal_acc:.1f}% cMAP {val_cmap:.3f} Gap {gap:.1f}%"
        )
        log_to_file(exp_dir, log_msg)
        print(log_msg)

        save_checkpoint(exp_dir, model, optimizer, epoch, val_top1)

        # Early stopping & best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            save_checkpoint(exp_dir, model, optimizer, epoch, val_top1, is_best=True)
            log_to_file(exp_dir, f"🏆 NEW BEST: {val_top1:.1f}%")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["training"]["early_stop_patience"]:
            log_to_file(exp_dir, f"EARLY STOPPING at Epoch {epoch + 1}")
            break

    metrics_df = pd.DataFrame(metrics_list)
    save_metrics(exp_dir, metrics_df)

    final_msg = f"EFFICIENTNET-B3 FINAL BEST: {best_val_acc:.1f}% → {exp_dir / 'checkpoints'}"
    log_to_file(exp_dir, final_msg)
    print("\n" + "=" * 90)
    print("PRIOR PROTECTED! " + final_msg)
    print("EfficientNet-B3 COMPLETE!")
    print("=" * 90)

    export_to_onnx(exp_dir, model, device)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
