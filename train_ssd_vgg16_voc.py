#!/usr/bin/env python
"""
Train SSD300‑VGG16 on Pascal VOC 2007
Includes:
 - training + validation (loss + mAP@0.5)
 - checkpoint save/resume
 - plotting loss & mAP curves
"""

import os, argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import ssd300_vgg16
import matplotlib.pyplot as plt

from utils import (transform_300, collate_fn,
                   compute_map)

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default="",
    help="Checkpoint path to resume from")
args = parser.parse_args()

# --- Config ---
DATA_ROOT     = "VOCdevkit/"
YEAR          = "2007"
BATCH_SIZE    = 8
NUM_EPOCHS    = 10
LR            = 1e-4
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
CHECKPOINT_DIR= "checkpoints"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --- DataLoaders ---
train_ds = VOCDetection(DATA_ROOT, year=YEAR, image_set="train",
                        download=True, transform=transform_300)
val_ds   = VOCDetection(DATA_ROOT, year=YEAR, image_set="val",
                        download=True, transform=transform_300)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

# --- Model & Optimizer ---
model = ssd300_vgg16(weights=None, num_classes=21).to(DEVICE)
optim = torch.optim.SGD(model.parameters(), lr=LR,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# --- Checkpoint fns ---
def save_ckpt(epoch):
    path = f"{CHECKPOINT_DIR}/ssd_epoch_{epoch}.pth"
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict()}, path)
    print(f"[✓] Saved {path}")

def load_ckpt(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck["model"])
    optim.load_state_dict(ck["optim"])
    print(f"[✓] Loaded '{path}', resuming at epoch {ck['epoch']+1}")
    return ck["epoch"] + 1

# --- Resume if requested ---
start_epoch = 0
if args.resume and os.path.isfile(args.resume):
    start_epoch = load_ckpt(args.resume)
elif args.resume:
    print(f"[!] No checkpoint at '{args.resume}', starting fresh")

# --- Train/Eval loop ---
train_losses, val_losses, val_maps = [], [], []

for epoch in range(start_epoch, NUM_EPOCHS):
    # train
    model.train()
    running = 0.0
    for imgs, tgts in train_loader:
        imgs = [i.to(DEVICE) for i in imgs]
        tgts = [{k: v.to(DEVICE) for k,v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        optim.zero_grad(); loss.backward(); optim.step()
        running += loss.item()
    train_loss = running / len(train_loader)

    # validation loss (forcing train-mode for loss)
    was_train = model.training
    model.train()
    val_running = 0.0
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs = [i.to(DEVICE) for i in imgs]
            tgts = [{k: v.to(DEVICE) for k,v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            val_running += sum(loss_dict.values()).item()
    if not was_train: model.eval()
    val_loss = val_running / len(val_loader)

    # mAP@0.5
    map50 = compute_map(model, val_loader, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_maps.append(map50)

    print(f"Epoch {epoch}/{NUM_EPOCHS-1} ▶ "
          f"train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"mAP@0.5: {map50:.4f}")

    save_ckpt(epoch)

# --- Save final weights ---
final_pth = f"{CHECKPOINT_DIR}/ssd_final.pth"
torch.save(model.state_dict(), final_pth)
print(f"[✓] Saved final weights: {final_pth}")

# --- Plot curves ---
epochs = list(range(start_epoch, NUM_EPOCHS))
plt.figure(); plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Loss Curves")
plt.savefig("outputs/loss_curve.png")
print("→ outputs/loss_curve.png")

plt.figure(); plt.plot(epochs, val_maps, marker='o', label="mAP@0.5")
plt.xlabel("Epoch"); plt.ylabel("mAP@0.5"); plt.legend()
plt.title("Validation mAP over Epochs")
plt.savefig("outputs/map_curve.png")
print("→ outputs/map_curve.png")
