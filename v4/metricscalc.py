import os
import cv2
import numpy as np
from dotenv import load_dotenv

# =========================
# FUNÇÕES
# =========================

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

def load_boxes(path, w, h, is_pred=False):
    boxes = []

    if not os.path.exists(path):
        return boxes

    with open(path) as f:
        for l in f:
            parts = list(map(float, l.split()))

            if is_pred:
                _, xc, yc, bw, bh, conf = parts
            else:
                _, xc, yc, bw, bh = parts

            boxes.append(yolo_to_xyxy([xc, yc, bw, bh], w, h))

    return boxes

# =========================
# CONFIG
# =========================

load_dotenv()

BASE_LOW  = os.getenv("BASE_LOW")
BASE_HIGH = os.getenv("BASE_HIGH")

IMG_DIR = os.path.join(BASE_LOW, "data3/images/test")  

GT_LOW_DIR   = os.path.join(BASE_LOW, "data3/labels/test")
GT_HIGH_DIR  = os.path.join(BASE_HIGH, "data3/labels/test")

PRED_LOW_DIR  = os.path.join(BASE_LOW, "data3/predictions/test/labels")
PRED_HIGH_DIR = os.path.join(BASE_HIGH, "data3/predictions/test/labels")

ious, dices = [], []

# =========================
# LOOP
# =========================

for img_name in os.listdir(IMG_DIR):

    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    # =========================
    # LOAD BOXES (LOW + HIGH)
    # =========================

    gt_boxes = []
    pred_boxes = []

    # LOW
    gt_boxes += load_boxes(
        os.path.join(GT_LOW_DIR, img_name.replace(".jpg", ".txt")),
        w, h
    )

    pred_boxes += load_boxes(
        os.path.join(PRED_LOW_DIR, img_name.replace(".jpg", ".txt")),
        w, h,
        is_pred=True
    )

    # HIGH
    gt_boxes += load_boxes(
        os.path.join(GT_HIGH_DIR, img_name.replace(".jpg", ".txt")),
        w, h
    )

    pred_boxes += load_boxes(
        os.path.join(PRED_HIGH_DIR, img_name.replace(".jpg", ".txt")),
        w, h,
        is_pred=True
    )

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        continue

    # =========================
    # MATCHING
    # =========================

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

# =========================
# RESULTADOS
# =========================

print(f"📊 IoU médio:  {np.mean(ious):.4f}")
print(f"📊 Dice médio: {np.mean(dices):.4f}")
print(f"📦 Total de matches: {len(ious)}")