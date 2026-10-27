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
    return np.array([x1, y1, x2, y2])

def intersection(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    return max(0, xB-xA) * max(0, yB-yA)

def area(a):
    return max(0, a[2]-a[0]) * max(0, a[3]-a[1])

def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0

def dice(a, b):
    inter = intersection(a, b)
    return 2 * inter / (area(a) + area(b)) if (area(a)+area(b)) > 0 else 0

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
IMG_DIR  = os.path.join(BASE_DIR, "data/images/test")
GT_DIR   = os.path.join(BASE_DIR, "data/labels/test")
PRED_DIR = os.path.join(BASE_DIR, "data/predictions/test/labels")

ious, dices = [], []

for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    gt_path   = os.path.join(GT_DIR, img_name.replace(".jpg", ".txt"))
    pred_path = os.path.join(PRED_DIR, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        continue

    # -------- LER GT --------
    gt_boxes = []
    with open(gt_path) as f:
        for l in f:
            _, xc, yc, bw, bh = map(float, l.split())
            gt_boxes.append(yolo_to_xyxy([xc, yc, bw, bh], w, h))

    # -------- LER PRED --------
    pred_boxes = []
    with open(pred_path) as f:
        for l in f:
            _, xc, yc, bw, bh, conf = map(float, l.split())
            pred_boxes.append(yolo_to_xyxy([xc, yc, bw, bh], w, h))

    matched_gt = set()

    for p in pred_boxes:
        best_iou = 0
        best_j = None

        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_j is not None:
            matched_gt.add(best_j)
            ious.append(best_iou)
            dices.append(dice(p, gt_boxes[best_j]))

# -------- RESULTADOS --------
print(f"📊 IoU médio:  {np.mean(ious):.4f}")
print(f"📊 Dice médio: {np.mean(dices):.4f}")
print(f"📦 Total de matches: {len(ious)}")
