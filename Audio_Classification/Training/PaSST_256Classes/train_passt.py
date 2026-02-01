import os
import time
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from hear21passt.base import get_basic_model

# Local Import
from dataset import MelDataset


# --- GLOBAL CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set your current phase here
PHASE = 2
MIXUP_ALPHA = 0.4 if PHASE == 2 else 0.0
USE_FOCAL_LOSS = False  # Set to True to enable Focal Loss instead of weighted CE

# One timestamp per run (ensures consistent folder names everywhere)
RUN_TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_ROOT = CURRENT_DIR / "current" / f"PaSST_{RUN_TIMESTAMP}"

CONFIG = {
    "paths": {
        "train_csv": CURRENT_DIR / "train.csv",
        "val_csv": CURRENT_DIR / "val.csv",
        "stats_json": CURRENT_DIR / "passt_stats.json",
        "class_map": CURRENT_DIR / "class_map.json",
        "class_weights": CURRENT_DIR / "class_weights.json",

        # ✅ ALL RUN ARTIFACTS LIVE UNDER: current/PaSST_<timestamp>/
        "run_root": RUN_ROOT,
        "checkpoint_dir": RUN_ROOT / "checkpoints",
        "log_dir": RUN_ROOT / "logs",
        "metrics_dir": RUN_ROOT / "metrics",

        # Phase-1 load path (kept global by default; change if you want it per-run)
        "load_checkpoint": CURRENT_DIR / "checkpoints" / "best_model_phase_1.pth",
    },
    "hyperparams": {
        "batch_size": 16,
        "epochs": 30 if PHASE == 2 else 20,
        "lr": 1e-5 if PHASE == 2 else 1e-4,
        "weight_decay": 0.0001,
        "label_smoothing": 0.1,
        "max_grad_norm": 1.0,
    },
}


# ============= HELPERS =============

def make_jsonable(obj):
    """
    Recursively convert Path objects inside dicts/lists/tuples to str
    so the structure becomes JSON-serializable.
    """
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(make_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ============= LOSS FUNCTIONS =============

class FocalLoss(nn.Module):
    """
    Focal Loss: addresses class imbalance by focusing on hard examples.
    L_focal = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter (0 = CE, 2 is recommended)
    alpha: class weights
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ============= MODEL UTILS =============

def get_passt_model(num_classes: int):
    """
    Initialize PaSST model for bird species classification.
    Removes internal mel conversion and replaces classification head.
    """
    print(f"Initializing PaSST for {num_classes} classes...")
    model = get_basic_model(mode="logits")

    # CRITICAL: Bypass internal audio-to-mel conversion
    # Your dataset already provides Mel spectrograms
    model.mel = nn.Identity()

    # Replace the classification head
    if isinstance(model.net.head, nn.Sequential):
        in_features = model.net.head[-1].in_features
        model.net.head[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.net.head.in_features
        model.net.head = nn.Linear(in_features, num_classes)

    return model


def mixup_data(x, y, alpha=0.4):
    """
    Mixup: create virtual training examples by interpolating between samples.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss as weighted combination of both labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def top_k_accuracy(outputs, targets, k=5):
    """
    Compute top-k accuracy.
    Args:
        outputs: [batch_size, num_classes] logits
        targets: [batch_size] ground truth labels
        k: top-k value (default 5)
    """
    _, pred = torch.topk(outputs, k, dim=1)
    expanded_targets = targets.view(-1, 1).expand_as(pred)
    correct = (pred == expanded_targets).sum().item()
    return correct


# ============= TRAINING ENGINE =============

def main():
    # 1. SETUP LOGGING AND PATHS (all under RUN_ROOT)
    for path_key in ["checkpoint_dir", "log_dir", "metrics_dir"]:
        CONFIG["paths"][path_key].mkdir(parents=True, exist_ok=True)

    # Paths for config and metrics JSON
    run_config_path = CONFIG["paths"]["metrics_dir"] / "run_config.json"
    json_metrics_path = CONFIG["paths"]["metrics_dir"] / f"{RUN_TIMESTAMP}_metrics.json"

    # Save CONFIG once at the very beginning (JSON-safe)
    with open(run_config_path, "w") as f:
        json.dump(make_jsonable(CONFIG), f, indent=4)

    with open(CONFIG["paths"]["class_map"]) as f:
        class_to_idx = json.load(f)
        num_classes = len(class_to_idx)

    # Load idx to class mapping for logging (if you ever need it)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # ✅ Use the same RUN_TIMESTAMP everywhere (consistent naming)
    run_name = f"phase_{PHASE}_{RUN_TIMESTAMP}"

    writer = SummaryWriter(CONFIG["paths"]["log_dir"] / run_name)
    csv_log_path = CONFIG["paths"]["log_dir"] / f"{run_name}_metrics.csv"

    # Initialize CSV with headers
    with open(csv_log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_top5_acc",
                "val_loss",
                "val_acc",
                "val_top5_acc",
                "lr",
            ]
        )

    # Initialize metrics storage (CONFIG is in a separate file already)
    metrics_history = {
        "phase": PHASE,
        "use_focal_loss": USE_FOCAL_LOSS,
        "run_timestamp": RUN_TIMESTAMP,
        "epochs": [],
    }

    # 2. MODEL INITIALIZATION
    torch.cuda.empty_cache()
    model = get_passt_model(num_classes).to(DEVICE)

    if PHASE == 2:
        ckpt_path = CONFIG["paths"]["load_checkpoint"]
        if ckpt_path.exists():
            print(f"✅ Loading Phase 1 weights from {ckpt_path}")
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE), strict=False
            )
        else:
            print("⚠️ No Phase 1 checkpoint found. Fine-tuning from base model.")

    # Freeze/Unfreeze Logic
    if PHASE == 1:
        print("PHASE 1: Training Head Only")
        for name, param in model.named_parameters():
            param.requires_grad = ("head" in name) or ("norm" in name)
    else:
        print("PHASE 2: Full Fine-Tuning (Memory-Optimized)")
        for param in model.parameters():
            param.requires_grad = True

    # 3. DATA LOADERS
    train_dataset = MelDataset(
        CONFIG["paths"]["train_csv"], CONFIG["paths"]["stats_json"], mode="train"
    )
    val_dataset = MelDataset(
        CONFIG["paths"]["val_csv"], CONFIG["paths"]["stats_json"], mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["hyperparams"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["hyperparams"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # 4. OPTIMIZER, SCHEDULER, CRITERION
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["hyperparams"]["lr"],
        weight_decay=CONFIG["hyperparams"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["hyperparams"]["lr"] * 5,
        total_steps=len(train_loader) * CONFIG["hyperparams"]["epochs"],
    )

    # Load class weights if available
    class_weights = None
    if CONFIG["paths"]["class_weights"].exists():
        with open(CONFIG["paths"]["class_weights"]) as f:
            weight_dict = json.load(f)
            class_weights = torch.tensor(
                [weight_dict[str(i)] for i in range(num_classes)],
                dtype=torch.float32,
                device=DEVICE,
            )
            print("✅ Loaded class weights from file")

    # Initialize loss function
    if USE_FOCAL_LOSS:
        print("🔥 Using Focal Loss with gamma=2.0")
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        print("📊 Using Weighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=CONFIG["hyperparams"]["label_smoothing"],
        )

    # 5. TRAINING LOOP WITH GRADIENT ACCUMULATION
    best_acc = 0.0
    best_top5_acc = 0.0
    global_step = 0
    accum_steps = 16 // CONFIG["hyperparams"]["batch_size"]

    print(f"\n{'='*70}")
    print(f"Run root: {CONFIG['paths']['run_root']}")
    print(f"Starting PHASE {PHASE} | Device: {DEVICE} | Using {num_classes} classes")
    print(f"{'='*70}\n")

    for epoch in range(CONFIG["hyperparams"]["epochs"]):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_top5_correct = 0
        train_total = 0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['hyperparams']['epochs']} [TRAIN]",
        )

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Dimension handling for PaSST
            if images.dim() > 3:
                images = images.view(images.size(0), 128, -1)

            if images.shape[-1] > 998:
                images = images[:, :, :998]

            # Mixup logic
            if PHASE == 2 and MIXUP_ALPHA > 0:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, MIXUP_ALPHA
                )
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss for accumulation
            loss = loss / accum_steps
            loss.backward()

            # Weight update step
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["hyperparams"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accum_steps

            # Top-1 accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Top-5 accuracy
            train_top5_correct += top_k_accuracy(outputs, labels, k=5)

            global_step += 1
            writer.add_scalar("Loss/Train_Batch", loss.item() * accum_steps, global_step)
            pbar.set_postfix(loss=f"{train_loss/(i+1):.4f}")

        train_acc = 100 * train_correct / train_total
        train_top5_acc = 100 * train_top5_correct / train_total

        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_top5_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar_val = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{CONFIG['hyperparams']['epochs']} [VAL]",
            )
            for images, labels in pbar_val:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                if images.dim() > 3:
                    images = images.view(images.size(0), 128, -1)
                if images.shape[-1] > 998:
                    images = images[:, :, :998]

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Top-5 accuracy
                val_top5_correct += top_k_accuracy(outputs, labels, k=5)

        val_acc = 100 * val_correct / val_total
        val_top5_acc = 100 * val_top5_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # ========== LOGGING ==========
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train_Top1", train_acc, epoch)
        writer.add_scalar("Accuracy/Train_Top5", train_top5_acc, epoch)
        writer.add_scalar("Accuracy/Val_Top1", val_acc, epoch)
        writer.add_scalar("Accuracy/Val_Top5", val_top5_acc, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # CSV logging
        with open(csv_log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    f"{avg_train_loss:.6f}",
                    f"{train_acc:.4f}",
                    f"{train_top5_acc:.4f}",
                    f"{avg_val_loss:.6f}",
                    f"{val_acc:.4f}",
                    f"{val_top5_acc:.4f}",
                    f"{current_lr:.2e}",
                ]
            )

        # JSON logging for each epoch (flush to disk every epoch)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "train_acc_top1": float(train_acc),
            "train_acc_top5": float(train_top5_acc),
            "val_loss": float(avg_val_loss),
            "val_acc_top1": float(val_acc),
            "val_acc_top5": float(val_top5_acc),
            "learning_rate": float(current_lr),
        }
        metrics_history["epochs"].append(epoch_metrics)

        with open(json_metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)

        # Console output
        print(f"\n📊 Epoch {epoch+1}/{CONFIG['hyperparams']['epochs']}")
        print(
            f"   Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5_acc:.2f}%"
        )
        print(
            f"   Val Loss:   {avg_val_loss:.6f} | Val Acc:   {val_acc:.2f}% | Val Top-5:   {val_top5_acc:.2f}%"
        )
        print(f"   LR: {current_lr:.2e}\n")

        # Checkpointing (based on top-1 accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            best_top5_acc = val_top5_acc
            save_path = (
                CONFIG["paths"]["checkpoint_dir"] / f"best_model_phase_{PHASE}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(
                f"🌟 New best Top-1 Acc: {val_acc:.2f}% | Top-5 Acc: {val_top5_acc:.2f}% (saved)"
            )

    print("\n✅ Training complete!")
    print(f"   Run root: {CONFIG['paths']['run_root']}")
    print(f"   Best Val Acc (Top-1): {best_acc:.2f}%")
    print(f"   Best Val Acc (Top-5): {best_top5_acc:.2f}%")
    print(f"   Metrics saved to: {json_metrics_path}")

    writer.close()


if __name__ == "__main__":
    main()
