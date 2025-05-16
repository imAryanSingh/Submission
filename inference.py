#!/usr/bin/env python
"""
Run demo inference on all images in test_images/,
draw and save annotated results in outputs/.
"""

import os, torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models.detection import ssd300_vgg16

from utils import transform_300, draw_boxes_on_original, CLASS_NAMES

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH   = "checkpoints/ssd_final.pth"
TEST_DIR    = "test_images"
OUT_DIR     = "outputs"
CONF_THRESH = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# load model
model = ssd300_vgg16(weights=None, num_classes=21).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

for fn in os.listdir(TEST_DIR):
    if not fn.lower().endswith((".jpg","png","jpeg")):
        continue

    img_path = os.path.join(TEST_DIR, fn)
    orig = Image.open(img_path).convert("RGB")
    inp = transform_300(orig).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(inp)[0]

    # draw on original
    out_img = draw_boxes_on_original(
        orig, pred["boxes"], pred["scores"], pred["labels"], thresh=CONF_THRESH
    )
    save_path = os.path.join(OUT_DIR, fn)
    out_img.save(save_path)
    print(f"[✓] Saved inference: {save_path}")
