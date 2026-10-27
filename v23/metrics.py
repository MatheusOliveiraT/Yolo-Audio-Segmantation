
import os
import cv2
import numpy as np
from dotenv import load_dotenv

def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    x1 = (xc - bw/2) * w
    y1 = (yc - bh/2) * h
    x2 = (xc + bw/2) * w
    y2 = (yc + bh/2) * h
    return [x1, y1, x2, y2]

def intersection(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    return max(0, xB-xA) * max(0, yB-yA)

def iou(a, b):
    inter = intersection(a, b)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def dice(a, b):
    inter = intersection(a, b)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return 2*inter / (areaA + areaB) if (areaA+areaB) > 0 else 0

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
IMG_DIR = os.path.join(BASE_DIR, "data/images/test")
GT_DIR = os.path.join(BASE_DIR, "data/labels/test")
PRED_DIR = os.path.join(BASE_DIR, "data/predictions/test/labels")

ious, dices = [], []

for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    gt_path = os.path.join(GT_DIR, img_name.replace(".jpg", ".txt"))
    pred_path = os.path.join(PRED_DIR, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        continue

    gt_boxes, pred_boxes = [], []

    with open(gt_path) as f:
        for l in f:
            _, xc, yc, bw, bh = map(float, l.split())
            gt_boxes.append(yolo_to_xyxy([xc, yc, bw, bh], w, h))

    with open(pred_path) as f:
        for l in f:
            _, xc, yc, bw, bh, conf = map(float, l.split())
            pred_boxes.append(yolo_to_xyxy([xc, yc, bw, bh], w, h))

    for p in pred_boxes:
        best_iou = 0
        best_gt = None
        for g in gt_boxes:
            v = iou(p, g)
            if v > best_iou:
                best_iou = v
                best_gt = g
        if best_gt is not None:
            ious.append(best_iou)
            dices.append(dice(p, best_gt))

print(f"IoU médio:  {np.mean(ious):.4f}")
print(f"Dice médio: {np.mean(dices):.4f}")
