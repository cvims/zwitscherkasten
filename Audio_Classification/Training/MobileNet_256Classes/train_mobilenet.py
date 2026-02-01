
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import os
import warnings
from sklearn.metrics import balanced_accuracy_score, average_precision_score

warnings.filterwarnings('ignore')

# 🆕 FIXED: CURRENT_DIR DEFINITION
CURRENT_DIR = Path(__file__).parent.resolve()

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_DIR = "training_mobilenet_antioverfit"

CONFIG = {
    "num_classes": 256,
    "data": {
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "preprocessed_mels_dir": "preprocessed_mels",
    },
    "training": {
        "batch_size": 8,
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "max_time_steps": 1024,
        "early_stop_patience": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}

# ============================================================================
# CREATE EXPERIMENT FOLDER
# ============================================================================

def setup_experiment_dir():
    exp_dir = Path(EXPERIMENT_DIR)
    exp_dir.mkdir(exist_ok=True)
    
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "onnx").mkdir(exist_ok=True)
    
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    
    print(f"📁 Experiment: {exp_dir.absolute()}")
    return exp_dir

# ============================================================================
# FIXED DATASET
# ============================================================================

class FixedMelDataset(Dataset):
    def __init__(self, csv_file, preprocessed_dir, max_time=1024, augment=False):
        self.df = pd.read_csv(csv_file)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_time = max_time
        self.augment = augment
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = self.preprocessed_dir / row['filepath']
        
        mel = np.load(filepath).astype(np.float32)
        if mel.shape[1] > self.max_time:
            mel = mel[:, :self.max_time]
        else:
            pad_width = self.max_time - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        
        mel = torch.from_numpy(mel / 255.0).unsqueeze(0)
        
        if self.augment:
            if torch.rand(1) < 0.5:
                t_mask = int(self.max_time * 0.15)
                t_start = torch.randint(0, self.max_time - t_mask, (1,)).item()
                mel[:, :, t_start:t_start + t_mask] *= 0.0
            if torch.rand(1) < 0.5:
                f_mask = 12
                f_start = torch.randint(0, 128 - f_mask, (1,)).item()
                mel[:, f_start:f_start + f_mask, :] *= 0.0
        
        label = int(row['label'])
        return mel, torch.tensor(label, dtype=torch.long)

# ============================================================================
# MODEL
# ============================================================================

class MobileNetV3Audio(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()
        base_model = models.mobilenet_v3_small(pretrained=False)
        base_model.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="Train")
    for batch_idx, (mels, labels) in enumerate(pbar):
        # 🆕 DEBUG: Check label range on first batch
        if batch_idx == 0:
            label_min = labels.min().item()
            label_max = labels.max().item()
            print(f"\n✓ First batch labels - Min: {label_min}, Max: {label_max}, Expected range: [0, {CONFIG['num_classes']-1}]")
            if label_min < 0 or label_max >= CONFIG['num_classes']:
                raise ValueError(f"❌ Labels out of range! Min={label_min}, Max={label_max}, but num_classes={CONFIG['num_classes']}")
        
        mels, labels = mels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        mem_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        pbar.set_postfix(loss=f'{loss:.3f}', acc=f'{100*correct/total:.1f}%', mem=f'{mem_gb:.1f}G')
    
    return total_loss / len(loader), 100 * correct / total

# 🆕 FIXED VALIDATE FUNCTION WITH BIOACOUSTIC METRICS
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
            
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            top5_correct += (logits.topk(5, 1)[1] == labels.unsqueeze(1)).any(1).sum().item()
            total += labels.size(0)
            
            # Collect for bioacoustic metrics (softmax probs)
            all_logits.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels.cpu())
    
    # Average loss
    avg_loss = total_loss / total
    
    # Standard accuracies
    top1_acc = 100 * correct / total
    top5_acc = 100 * top5_correct / total
    
    # Bioacoustic metrics
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    preds = all_logits.argmax(1)
    bal_acc = balanced_accuracy_score(all_labels, preds)
    
    # Class-wise macro Average Precision (cMAP)
    try:
        ap_scores = average_precision_score(all_labels, all_logits, average=None)
        cmap_macro = float(np.nanmean(ap_scores))
    except ValueError:
        cmap_macro = 0.0
    
    return (avg_loss, top1_acc, top5_acc, 100 * bal_acc, cmap_macro)

# ============================================================================
# SAVING FUNCTIONS
# ============================================================================

def save_checkpoint(exp_dir, model, optimizer, epoch, val_acc, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': CONFIG
    }
    
    filename = f"model_epoch{epoch+1:02d}_val{val_acc:.1f}.pth"
    torch.save(checkpoint, exp_dir / "checkpoints" / filename)
    
    if is_best:
        torch.save(checkpoint, exp_dir / "checkpoints" / "BEST_model.pth")

def save_metrics(exp_dir, metrics_df):
    csv_path = exp_dir / "metrics" / "training_metrics.csv"
    json_path = exp_dir / "metrics" / "training_metrics.json"
    
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_json(json_path, orient='records', indent=2)

def log_to_file(exp_dir, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(exp_dir / "logs" / "training.log", 'a') as f:
        f.write(log_entry)
    print(log_entry.strip())

# ============================================================================
# 🆕 ONNX EXPORT FUNCTION
# ============================================================================

def export_to_onnx(exp_dir, model, device, input_shape=(1, 1, 128, 1024)):
    """
    Export trained model to ONNX format
    
    Args:
        exp_dir: Experiment directory
        model: PyTorch model
        device: Device (cpu/cuda)
        input_shape: (batch_size, channels, height, width) for mel spectrograms
    """
    
    onnx_dir = exp_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*90)
    print("🔄 Exporting model to ONNX format...")
    print("="*90)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Export paths
    onnx_path = onnx_dir / "mobilenet_v3_audio.onnx"
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['mel_spectrogram'],
        output_names=['logits'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size', 3: 'time_steps'},
            'logits': {0: 'batch_size'}
        },
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
    )
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    
    # Save metadata
    metadata = {
        'model_name': 'MobileNetV3Audio',
        'num_classes': CONFIG['num_classes'],
        'input_shape': list(input_shape),
        'input_description': 'Mel spectrogram (batch, 1 channel, 128 freq bins, 1024 time steps)',
        'output_description': f'Logits for {CONFIG["num_classes"]} bird species',
        'expected_input_range': [0.0, 1.0],
        'preprocessing': 'Normalize mel by dividing by 255',
        'export_date': datetime.now().isoformat(),
    }
    
    metadata_path = onnx_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Save requirements for ONNX inference
    requirements_onnx = """# Requirements for ONNX inference
onnx==1.15.0
onnxruntime==1.16.3
numpy>=1.21.0
"""
    
    req_path = onnx_dir / "requirements_onnx.txt"
    with open(req_path, 'w') as f:
        f.write(requirements_onnx)
    
    print(f"✅ ONNX requirements saved to: {req_path}")
    
    # Create inference example script
    inference_script = '''#!/usr/bin/env python3
"""
Example inference script using ONNX model
"""

import onnxruntime as ort
import numpy as np
import json
from pathlib import Path

# Load ONNX model
model_path = Path(__file__).parent / "mobilenet_v3_audio.onnx"
session = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Example: Create dummy mel spectrogram
mel_spec = np.random.rand(1, 1, 128, 1024).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

logits = session.run([output_name], {input_name: mel_spec})[0]

# Get prediction
pred_class = np.argmax(logits, axis=1)[0]
confidence = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
pred_confidence = confidence[pred_class]

print(f"Predicted class: {pred_class}")
print(f"Confidence: {pred_confidence:.4f}")
print(f"Logits shape: {logits.shape}")
'''
    
    inference_path = onnx_dir / "inference_example.py"
    with open(inference_path, 'w') as f:
        f.write(inference_script)
    
    print(f"✅ Inference example saved to: {inference_path}")
    print("\n" + "="*90)
    print("🎉 ONNX export complete!")
    print("="*90 + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    exp_dir = setup_experiment_dir()
    log_to_file(exp_dir, "🚀 Starting MobileNetV3 Anti-Overfit Training")
    
    device = torch.device(CONFIG["training"]["device"])
    
    # 🆕 FIXED: Use CURRENT_DIR for data paths
    train_dataset = FixedMelDataset(
        str(CURRENT_DIR / CONFIG["data"]["train_csv"]),
        str(CURRENT_DIR / CONFIG["data"]["preprocessed_mels_dir"]),
        CONFIG["training"]["max_time_steps"], augment=True
    )
    val_dataset = FixedMelDataset(
        str(CURRENT_DIR / CONFIG["data"]["val_csv"]),
        str(CURRENT_DIR / CONFIG["data"]["preprocessed_mels_dir"]), 
        CONFIG["training"]["max_time_steps"], augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"],
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"],
                          shuffle=False, num_workers=4, pin_memory=True)
    
    log_to_file(exp_dir, f"✅ Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    model = MobileNetV3Audio(CONFIG["num_classes"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log_to_file(exp_dir, f"✅ Model: {total_params:,} params")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=CONFIG["training"]["learning_rate"],
                          weight_decay=CONFIG["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    metrics_list = []
    
    log_to_file(exp_dir, "🚀 Training gestartet...")
    
    for epoch in range(CONFIG["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['training']['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 🆕 FIXED: Unpack all 5 return values
        val_loss, val_top1, val_top5, val_bal_acc, val_cmap = validate(model, val_loader, criterion, device)
        
        # 🆕 FIXED: Use val_top1 instead of undefined val_acc
        scheduler.step(val_top1)
        
        gap = train_acc - val_top1
        mem_used = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
        
        # 🆕 UPDATED: Include bioacoustic metrics
        metrics = {
            'epoch': epoch + 1, 
            'train_loss': train_loss, 
            'train_acc': train_acc,
            'val_loss': val_loss, 
            'val_top1': val_top1, 
            'val_top5': val_top5,
            'val_bal_acc': val_bal_acc,
            'val_cmap': val_cmap,
            'train_val_gap': gap, 
            'lr': optimizer.param_groups[0]['lr'], 
            'memory_gb': mem_used
        }
        metrics_list.append(metrics)
        
        # 🆕 IMPROVED: Better logging with bioacoustic metrics
        log_msg = f"Epoch {epoch+1}: Train {train_acc:.1f}% | Val Top1 {val_top1:.1f}% Top5 {val_top5:.1f}% | BalAcc {val_bal_acc:.1f}% cMAP {val_cmap:.1f}% | Gap {gap:.1f}%"
        log_to_file(exp_dir, log_msg)
        print(log_msg)
        
        # SAVE EVERY EPOCH
        save_checkpoint(exp_dir, model, optimizer, epoch, val_top1)
        
        # EARLY STOPPING
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            save_checkpoint(exp_dir, model, optimizer, epoch, val_top1, is_best=True)
            log_to_file(exp_dir, f"🌟 NEW BEST: {val_top1:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["training"]["early_stop_patience"]:
                log_to_file(exp_dir, f"🛑 EARLY STOPPING at Epoch {epoch+1}")
                break
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics_list)
        save_metrics(exp_dir, metrics_df)
    
    final_msg = f"🏆 FINAL BEST: {best_val_acc:.1f}% | {exp_dir}/checkpoints/"
    log_to_file(exp_dir, final_msg)
    print("\n" + "="*90)
    print(final_msg)
    print("="*90)
    
    # 🆕 EXPORT TO ONNX
    export_to_onnx(exp_dir, model, device)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
