# utils.py

import os
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, ToTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# VOC class names
CLASS_NAMES = [
    "__background__", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Common transform: resize to 300×300 + to tensor
transform_300 = Compose([
    Resize((300, 300)),
    ToTensor(),
])

def collate_fn(batch):
    images, targets = [], []
    for img, anno in batch:
        images.append(img)
        objs = anno["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]
        boxes, labels = [], []
        for o in objs:
            bb = o["bndbox"]
            boxes.append([
                float(bb["xmin"]), float(bb["ymin"]),
                float(bb["xmax"]), float(bb["ymax"])
            ])
            labels.append(CLASS_TO_IDX[o["name"]])
        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        })
    return images, targets

def compute_map(model, loader, device):
    """Return mAP@0.5 on loader."""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            # format
            preds_fmt = [
                {"boxes": p["boxes"].cpu(),
                 "scores": p["scores"].cpu(),
                 "labels": p["labels"].cpu()}
                for p in preds
            ]
            tgts_fmt = [
                {"boxes":  t["boxes"].cpu(),
                 "labels": t["labels"].cpu()}
                for t in targets
            ]
            metric.update(preds_fmt, tgts_fmt)
    return metric.compute()["map_50"].item()

def draw_boxes_on_original(img: Image.Image, boxes, scores, labels, thresh=0.5):
    """Draw predicted boxes (300×300 coords) onto original image by rescaling."""
    orig_w, orig_h = img.size
    scale_x, scale_y = orig_w / 300, orig_h / 300
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        if score < thresh:
            continue
        x0, y0, x1, y1 = box.cpu().tolist()
        # rescale
        x0, y0, x1, y1 = x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), f"{CLASS_NAMES[label]} {score:.2f}", fill="red")
    return img
