from ultralytics import YOLO
from sklearn.model_selection import KFold
import os
import glob
import random
import shutil
import yaml
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
import numpy as np

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = "/home/a_matheusteixeira/yolo_dataset_old_cv"

N_SPLITS = 5
TEST_SPLIT = 0.20
RANDOM_STATE = 123

# ==========================================================
# LISTA DE IMAGENS
# ==========================================================

images = sorted(glob.glob(os.path.join(BASE_DIR, "images", "*.jpg")))

random.seed(RANDOM_STATE)
random.shuffle(images)

# ==========================================================
# METRICAS
# ==========================================================

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

# ==========================================================
# SEPARAÇÃO DO TESTE
# ==========================================================

n_test = int(len(images) * TEST_SPLIT)

test_images = images[:n_test]
cv_images = images[n_test:]

print(f"Total imagens : {len(images)}")
print(f"Teste fixo    : {len(test_images)}")
print(f"Treino/Val    : {len(cv_images)}")

train_txt = os.path.join(BASE_DIR, "data", "train_final.txt")

with open(train_txt, "w") as f:
    for img in cv_images:
        f.write(os.path.abspath(img) + "\n")

print(f"Arquivo criado: {train_txt}")

# ==========================================================
# SALVA O TESTE FIXO
# ==========================================================

test_dir = os.path.join(BASE_DIR, "data", "test")

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

# ==========================================================
# FUNÇÃO PARA COPIAR DADOS
# ==========================================================


def copy_split(img_list, split_name, base_dir):

    img_dir = os.path.join(base_dir, "images", split_name)
    lbl_dir = os.path.join(base_dir, "labels", split_name)

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path in img_list:

        base = os.path.basename(img_path)

        lbl_path = (
            img_path
            .replace("/images/", "/labels/")
            .replace(".jpg", ".txt")
        )

        shutil.copy(img_path, os.path.join(img_dir, base))

        if os.path.exists(lbl_path):
            shutil.copy(
                lbl_path,
                os.path.join(lbl_dir, base.replace(".jpg", ".txt"))
            )

copy_split(test_images, "test", os.path.join(BASE_DIR, "data"))

# ==========================================================
# KFOLD
# ==========================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

# ==========================================================
# LOOP DOS FOLDS
# ==========================================================

for fold, (train_idx, val_idx) in enumerate(kf.split(cv_images), start=1):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    train_images = [cv_images[i] for i in train_idx]
    val_images = [cv_images[i] for i in val_idx]

    print(f"Treino    : {len(train_images)}")
    print(f"Validação : {len(val_images)}")

    fold_dir = os.path.join(BASE_DIR, "data", f"fold_{fold}")

    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    copy_split(train_images, "train", fold_dir)
    copy_split(val_images, "val", fold_dir)

    # ======================================================
    # YAML
    # ======================================================

    data_yaml = {
        "path": fold_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["interest"]
    }

    yaml_path = os.path.join(fold_dir, "dataset.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    # ======================================================
    # TREINAMENTO
    # ======================================================

    model = YOLO("yolo26m.pt")

    model.train(
        data=yaml_path,
        epochs=200,
        imgsz=640,
        batch=128,
        patience=20,
        workers=4,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        pretrained=True,
        project=os.path.join(BASE_DIR, "runs"),
        name=f"fold_{fold}"
    )

    # ======================================================
    # VALIDAÇÃO
    # ======================================================

    metrics = model.val(
        data=yaml_path,
        split="val"
    )

    # ======================================================
    # IoU e Dice médios no conjunto de validação
    # ======================================================

    predictions = model.predict(
        source=os.path.join(fold_dir, "images", "val"),
        conf=0.25,
        save=False,
        verbose=False
    )

    ious = []
    dices = []

    for pred in predictions:

        img_path = pred.path

        img = Image.open(img_path)
        width, height = img.size

        # ---------- Ground Truth ----------
        label_path = (
            img_path
            .replace("/images/", "/labels/")
            .replace(".jpg", ".txt")
            .replace(".png", ".txt")
        )

        gt_boxes = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    values = list(map(float, line.strip().split()))

                    # classe x y w h
                    _, xc, yc, bw, bh = values

                    gt_boxes.append(
                        yolo_to_xyxy(
                            [xc, yc, bw, bh],
                            width,
                            height
                        )
                    )

        # ---------- Predições ----------
        pred_boxes = []

        if pred.boxes is not None:

            for box in pred.boxes.xyxy.cpu().numpy():
                pred_boxes.append(box.tolist())

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        # ---------- Hungarian Matching ----------

        cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)))

        for i, gt in enumerate(gt_boxes):
            for j, pb in enumerate(pred_boxes):
                cost_matrix[i, j] = 1 - iou(gt, pb)

        rows, cols = linear_sum_assignment(cost_matrix)

        for r, c in zip(rows, cols):

            current_iou = iou(gt_boxes[r], pred_boxes[c])
            current_dice = dice(gt_boxes[r], pred_boxes[c])

            ious.append(current_iou)
            dices.append(current_dice)

    mean_iou = np.mean(ious) if len(ious) else 0
    mean_dice = np.mean(dices) if len(dices) else 0

    # ======================================================
    # Salvar resultados
    # ======================================================

    results.append({
        "fold": fold,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "IoU": mean_iou,
        "Dice": mean_dice
    })
# ==========================================================
# RESULTADOS DA CROSS VALIDATION
# ==========================================================

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("RESULTADOS DA CROSS VALIDATION")
print("=" * 60)

print(df)

print("\nMédia")
print(df.mean(numeric_only=True))

print("\nDesvio padrão")
print(df.std(numeric_only=True))

df.to_csv(
    os.path.join(BASE_DIR, "cross_validation_results.csv"),
    index=False
)

print("\nResultados salvos.")