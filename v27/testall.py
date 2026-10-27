from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd
import shutil
from dotenv import load_dotenv

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
DATA_DIR = os.path.join(BASE_DIR, "data")

IMG_DIR = os.path.join(
    DATA_DIR,
    "images",
    "test"
)

GT_DIR = os.path.join(
    DATA_DIR,
    "labels",
    "test"
)

PRED_BASE_DIR = os.path.join(
    DATA_DIR,
    "predictions"
)

N_FOLDS = 5


# ==========================================================
# FUNÇÕES DE MÉTRICAS
# ==========================================================

def yolo_to_xyxy(box, w, h):

    xc, yc, bw, bh = box

    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h

    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h

    return [x1, y1, x2, y2]


def intersection(a, b):

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])

    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    return max(0, xB - xA) * max(0, yB - yA)


def iou(a, b):

    inter = intersection(a, b)

    areaA = (
        (a[2] - a[0]) *
        (a[3] - a[1])
    )

    areaB = (
        (b[2] - b[0]) *
        (b[3] - b[1])
    )

    union = areaA + areaB - inter

    return (
        inter / union
        if union > 0
        else 0
    )


def dice(a, b):

    inter = intersection(a, b)

    areaA = (
        (a[2] - a[0]) *
        (a[3] - a[1])
    )

    areaB = (
        (b[2] - b[0]) *
        (b[3] - b[1])
    )

    total = areaA + areaB

    return (
        2 * inter / total
        if total > 0
        else 0
    )


# ==========================================================
# FUNÇÃO PARA CALCULAR IoU/DICE
# ==========================================================

def calculate_metrics():

    ious = []
    dices = []

    # ------------------------------------------------------
    # Percorre todas as imagens do teste
    # ------------------------------------------------------

    image_names = sorted(
        os.listdir(IMG_DIR)
    )

    for img_name in image_names:

        # Ignora arquivos que não sejam imagens
        if not img_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        img_path = os.path.join(
            IMG_DIR,
            img_name
        )

        img = cv2.imread(img_path)

        if img is None:
            print(
                f"⚠ Não foi possível ler: {img_name}"
            )
            continue

        h, w = img.shape[:2]

        # --------------------------------------------------
        # Caminhos dos labels
        # --------------------------------------------------

        base_name = os.path.splitext(
            img_name
        )[0]

        gt_path = os.path.join(
            GT_DIR,
            base_name + ".txt"
        )

        pred_path = os.path.join(
            CURRENT_PRED_DIR,
            base_name + ".txt"
        )

        # --------------------------------------------------
        # Sem GT
        # --------------------------------------------------

        if not os.path.exists(gt_path):
            continue

        # --------------------------------------------------
        # Sem predição
        # --------------------------------------------------

        if not os.path.exists(pred_path):
            continue

        # --------------------------------------------------
        # Ground Truth
        # --------------------------------------------------

        gt_boxes = []

        with open(gt_path) as f:

            for line in f:

                values = line.strip().split()

                if len(values) < 5:
                    continue

                _, xc, yc, bw, bh = map(
                    float,
                    values[:5]
                )

                gt_boxes.append(
                    yolo_to_xyxy(
                        [xc, yc, bw, bh],
                        w,
                        h
                    )
                )

        # --------------------------------------------------
        # Predições
        # --------------------------------------------------

        pred_boxes = []

        with open(pred_path) as f:

            for line in f:

                values = line.strip().split()

                if len(values) < 5:
                    continue

                # YOLO prediction:
                # classe x y w h conf

                _, xc, yc, bw, bh = map(
                    float,
                    values[:5]
                )

                pred_boxes.append(
                    yolo_to_xyxy(
                        [xc, yc, bw, bh],
                        w,
                        h
                    )
                )

        # --------------------------------------------------
        # Se não houver GT ou predição
        # --------------------------------------------------

        if (
            len(gt_boxes) == 0
            or len(pred_boxes) == 0
        ):
            continue

        # --------------------------------------------------
        # Matching:
        # cada predição encontra o GT de maior IoU
        # --------------------------------------------------

        for pred_box in pred_boxes:

            best_iou = 0
            best_gt = None

            for gt_box in gt_boxes:

                current_iou = iou(
                    pred_box,
                    gt_box
                )

                if current_iou > best_iou:

                    best_iou = current_iou
                    best_gt = gt_box

            if best_gt is not None:

                ious.append(
                    best_iou
                )

                dices.append(
                    dice(
                        pred_box,
                        best_gt
                    )
                )

    # ======================================================
    # RESULTADOS
    # ======================================================

    mean_iou = (
        np.mean(ious)
        if len(ious) > 0
        else 0
    )

    std_iou = (
        np.std(ious)
        if len(ious) > 0
        else 0
    )

    mean_dice = (
        np.mean(dices)
        if len(dices) > 0
        else 0
    )

    std_dice = (
        np.std(dices)
        if len(dices) > 0
        else 0
    )

    return {
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "n_iou": len(ious),
        "n_dice": len(dices)
    }


# ==========================================================
# TESTAR TODOS OS FOLDS
# ==========================================================

results = []


for fold in range(1, N_FOLDS + 1):

    print("\n" + "=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    # ------------------------------------------------------
    # Modelo
    # ------------------------------------------------------

    model_path = os.path.join(
        BASE_DIR,
        "runs",
        f"fold_{fold}",
        "weights",
        "best.pt"
    )

    if not os.path.exists(model_path):

        print(
            f"❌ Modelo não encontrado:"
        )

        print(model_path)

        continue

    print(
        f"Modelo: {model_path}"
    )

    model = YOLO(model_path)

    # ------------------------------------------------------
    # Diretório específico do fold
    # ------------------------------------------------------

    fold_pred_dir = os.path.join(
        PRED_BASE_DIR,
        f"fold_{fold}"
    )

    # Remove predições anteriores desse fold
    if os.path.exists(fold_pred_dir):
        shutil.rmtree(fold_pred_dir)

    os.makedirs(
        fold_pred_dir,
        exist_ok=True
    )

    # ------------------------------------------------------
    # PREDIÇÕES
    # ------------------------------------------------------

    print("\nGerando predições no teste...")

    predictions = model.predict(

        source=IMG_DIR,

        imgsz=640,

        conf=0.25,

        save=False,

        save_txt=True,

        save_conf=True,

        project=PRED_BASE_DIR,

        name=f"fold_{fold}",

        exist_ok=True,

        verbose=False
    )

    print(
        f"Predições geradas: {len(predictions)}"
    )

    # ------------------------------------------------------
    # O Ultralytics cria:
    #
    # data/predictions/fold_X/
    #     labels/
    #
    # ------------------------------------------------------

    CURRENT_PRED_DIR = os.path.join(
        PRED_BASE_DIR,
        f"fold_{fold}",
        "labels"
    )

    if not os.path.exists(
        CURRENT_PRED_DIR
    ):

        print(
            "❌ Diretório de labels das "
            "predições não encontrado:"
        )

        print(CURRENT_PRED_DIR)

        continue

    # ------------------------------------------------------
    # CALCULA MÉTRICAS
    # ------------------------------------------------------

    metrics = calculate_metrics()

    print("\nResultados do fold:")

    print(
        f"IoU médio  : {metrics['mean_iou']:.4f}"
    )

    print(
        f"IoU std    : {metrics['std_iou']:.4f}"
    )

    print(
        f"Dice médio : {metrics['mean_dice']:.4f}"
    )

    print(
        f"Dice std   : {metrics['std_dice']:.4f}"
    )

    print(
        f"Matches IoU/Dice: {metrics['n_iou']}"
    )

    # ------------------------------------------------------
    # Salva resultado
    # ------------------------------------------------------

    results.append({

        "fold": fold,

        "IoU_mean": metrics["mean_iou"],

        "IoU_std": metrics["std_iou"],

        "Dice_mean": metrics["mean_dice"],

        "Dice_std": metrics["std_dice"],

        "matches": metrics["n_iou"]
    })


# ==========================================================
# RESULTADOS GERAIS
# ==========================================================

if len(results) == 0:

    print(
        "\n❌ Nenhum fold foi processado."
    )

    exit()


df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("RESULTADOS POR FOLD")
print("=" * 70)

print(
    df.to_string(index=False)
)


# ==========================================================
# MÉDIA E DESVIO PADRÃO ENTRE OS FOLDS
# ==========================================================

mean_iou_folds = df["IoU_mean"].mean()
std_iou_folds = df["IoU_mean"].std()

mean_dice_folds = df["Dice_mean"].mean()
std_dice_folds = df["Dice_mean"].std()


print("\n" + "=" * 70)
print("MÉDIA E DESVIO PADRÃO ENTRE OS FOLDS")
print("=" * 70)

print(
    f"IoU médio dos folds  : "
    f"{mean_iou_folds:.4f}"
)

print(
    f"IoU desvio padrão    : "
    f"{std_iou_folds:.4f}"
)

print(
    f"Dice médio dos folds : "
    f"{mean_dice_folds:.4f}"
)

print(
    f"Dice desvio padrão   : "
    f"{std_dice_folds:.4f}"
)


# ==========================================================
# MELHOR FOLD
# ==========================================================

best_iou_idx = df["IoU_mean"].idxmax()

best_dice_idx = df["Dice_mean"].idxmax()

best_iou_fold = df.loc[
    best_iou_idx,
    "fold"
]

best_iou_value = df.loc[
    best_iou_idx,
    "IoU_mean"
]

best_dice_fold = df.loc[
    best_dice_idx,
    "fold"
]

best_dice_value = df.loc[
    best_dice_idx,
    "Dice_mean"
]


print("\n" + "=" * 70)
print("MELHORES RESULTADOS")
print("=" * 70)

print(
    f"Melhor IoU  : Fold {int(best_iou_fold)} "
    f"({best_iou_value:.4f})"
)

print(
    f"Melhor Dice : Fold {int(best_dice_fold)} "
    f"({best_dice_value:.4f})"
)


# ==========================================================
# SALVA CSV
# ==========================================================

results_path = os.path.join(
    BASE_DIR,
    "test_folds_results.csv"
)

df.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 70)

print(
    f"Resultados salvos em:"
)

print(results_path)

print("=" * 70)
