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
from dotenv import load_dotenv

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")

N_SPLITS = 5
RANDOM_STATE = 42

# ==========================================================
# ÁUDIOS EXCLUSIVOS DO CONJUNTO DE TESTE
# ==========================================================

TEST_AUDIO_IDS = [
    "W04869764S2026516_20231218_084000",
    "W04821899S2014959_20231104_084000",
    "W04822429S2014229_20231116_065000",
    "W04821899S2014959_20231109_065000",
    "W04857290S2011661_20231130_073000",
    "W04870903S2027082_20231112_060000",
    "W04831109S2013814_20231111_050000",
    "W04822980S2014203_20231112_051000",
    "W04831846S2012265_20240228_061000",
    "W04870903S2027082_20240103_074000",
    "W04870903S2027082_20231217_072000",
    "W04870903S2027082_20231126_055000",
    "W04870894S2027093_20230813_131000",
    "W04857290S2011661_20231110_054000",
    "W04857176S2013800_20240605_085000",
    "W04857176S2013800_20240604_070000",
    "W04857176S2013800_20240429_073000",
    "W04823186S2014503_20240402_072000",
    "W04823186S2014503_20240302_075000",
    "W04822980S2014203_20240318_081000",
]

# ==========================================================
# LISTA DE IMAGENS
# ==========================================================

images = sorted(
    glob.glob(
        os.path.join(BASE_DIR, "images", "*.jpg")
    )
)

print(f"Total de imagens encontradas: {len(images)}")

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
# SEPARAÇÃO DO TESTE POR ÁUDIO
# ==========================================================

test_images = []
cv_images = []

for img_path in images:

    filename = os.path.basename(img_path)

    # Remove extensão
    image_id = os.path.splitext(filename)[0]

    # Verifica se a imagem pertence a algum dos áudios de teste
    is_test = any(
        image_id.startswith(audio_id)
        for audio_id in TEST_AUDIO_IDS
    )

    if is_test:
        test_images.append(img_path)
    else:
        cv_images.append(img_path)

# ==========================================================
# EMBARALHAMENTO APENAS DO CONJUNTO DE CV
# ==========================================================

random.seed(RANDOM_STATE)
random.shuffle(cv_images)

# ==========================================================
# INFORMAÇÕES DA SEPARAÇÃO
# ==========================================================

print("\n" + "=" * 60)
print("SEPARAÇÃO DO DATASET")
print("=" * 60)

print(f"Total de imagens : {len(images)}")
print(f"Teste fixo      : {len(test_images)}")
print(f"Treino/Val      : {len(cv_images)}")

print("\nÁudios utilizados exclusivamente no teste:")
for audio_id in TEST_AUDIO_IDS:

    audio_imgs = [
        img for img in test_images
        if os.path.splitext(os.path.basename(img))[0].startswith(audio_id)
    ]

    print(
        f"{audio_id}: "
        f"{len(audio_imgs)} imagens"
    )

# ==========================================================
# VERIFICA SE TODOS OS ÁUDIOS FORAM ENCONTRADOS
# ==========================================================

found_audio_ids = set()

for img_path in test_images:

    image_id = os.path.splitext(
        os.path.basename(img_path)
    )[0]

    for audio_id in TEST_AUDIO_IDS:

        if image_id.startswith(audio_id):
            found_audio_ids.add(audio_id)

missing_audio_ids = [
    audio_id
    for audio_id in TEST_AUDIO_IDS
    if audio_id not in found_audio_ids
]

if missing_audio_ids:

    print("\n" + "!" * 60)
    print("ATENÇÃO: ÁUDIOS NÃO ENCONTRADOS")
    print("!" * 60)

    for audio_id in missing_audio_ids:
        print(audio_id)

    print("\nVerifique se os nomes dos arquivos/imagens estão corretos.")

else:

    print("\nTodos os 20 áudios foram encontrados.")

# ==========================================================
# ARQUIVO COM OS DADOS DE TREINO/VAL
# ==========================================================

train_txt = os.path.join(
    BASE_DIR,
    "data",
    "train_final.txt"
)

os.makedirs(
    os.path.dirname(train_txt),
    exist_ok=True
)

with open(train_txt, "w") as f:

    for img in cv_images:
        f.write(os.path.abspath(img) + "\n")

print(f"\nArquivo criado: {train_txt}")

# ==========================================================
# SALVA O TESTE FIXO
# ==========================================================

test_dir = os.path.join(
    BASE_DIR,
    "data",
    "test"
)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

# ==========================================================
# FUNÇÃO PARA COPIAR DADOS
# ==========================================================

def copy_split(img_list, split_name, base_dir):

    img_dir = os.path.join(
        base_dir,
        "images",
        split_name
    )

    lbl_dir = os.path.join(
        base_dir,
        "labels",
        split_name
    )

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path in img_list:

        base = os.path.basename(img_path)

        # Caminho esperado da label original
        lbl_path = (
            img_path
            .replace("/images/", "/labels/")
            .replace(".jpg", ".txt")
        )

        # Copia imagem
        shutil.copy(
            img_path,
            os.path.join(img_dir, base)
        )

        # Copia label, caso exista
        if os.path.exists(lbl_path):

            shutil.copy(
                lbl_path,
                os.path.join(
                    lbl_dir,
                    base.replace(".jpg", ".txt")
                )
            )

# ==========================================================
# COPIA TESTE
# ==========================================================

copy_split(
    test_images,
    "test",
    os.path.join(BASE_DIR, "data")
)

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

for fold, (train_idx, val_idx) in enumerate(
    kf.split(cv_images),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    train_images = [
        cv_images[i]
        for i in train_idx
    ]

    val_images = [
        cv_images[i]
        for i in val_idx
    ]

    print(f"Treino    : {len(train_images)}")
    print(f"Validação : {len(val_images)}")

    fold_dir = os.path.join(
        BASE_DIR,
        "data",
        f"fold_{fold}"
    )

    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    copy_split(
        train_images,
        "train",
        fold_dir
    )

    copy_split(
        val_images,
        "val",
        fold_dir
    )

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

    yaml_path = os.path.join(
        fold_dir,
        "dataset.yaml"
    )

    with open(yaml_path, "w") as f:
        yaml.dump(
            data_yaml,
            f,
            sort_keys=False
        )

    # ======================================================
    # TREINAMENTO
    # ======================================================

    model = YOLO("yolo26m.pt")

    model.train(
        data=yaml_path,
        epochs=200,
        imgsz=640,
        batch=64,
        patience=20,
        workers=4,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        pretrained=True,
        project=os.path.join(
            BASE_DIR,
            "runs"
        ),
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
    # PREDIÇÕES NA VALIDAÇÃO
    # ======================================================

    predictions = model.predict(
        source=os.path.join(
            fold_dir,
            "images",
            "val"
        ),
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

        # --------------------------------------------------
        # Ground Truth
        # --------------------------------------------------

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

                    values = list(
                        map(
                            float,
                            line.strip().split()
                        )
                    )

                    # classe x y w h
                    _, xc, yc, bw, bh = values

                    gt_boxes.append(
                        yolo_to_xyxy(
                            [xc, yc, bw, bh],
                            width,
                            height
                        )
                    )

        # --------------------------------------------------
        # Predições
        # --------------------------------------------------

        pred_boxes = []

        if pred.boxes is not None:

            for box in (
                pred.boxes.xyxy
                .cpu()
                .numpy()
            ):

                pred_boxes.append(
                    box.tolist()
                )

        if (
            len(gt_boxes) == 0
            or len(pred_boxes) == 0
        ):
            continue

        # --------------------------------------------------
        # Hungarian Matching
        # --------------------------------------------------

        cost_matrix = np.ones(
            (
                len(gt_boxes),
                len(pred_boxes)
            )
        )

        for i, gt in enumerate(gt_boxes):

            for j, pb in enumerate(pred_boxes):

                cost_matrix[i, j] = (
                    1 - iou(gt, pb)
                )

        rows, cols = linear_sum_assignment(
            cost_matrix
        )

        for r, c in zip(rows, cols):

            current_iou = iou(
                gt_boxes[r],
                pred_boxes[c]
            )

            current_dice = dice(
                gt_boxes[r],
                pred_boxes[c]
            )

            ious.append(current_iou)
            dices.append(current_dice)

    # ======================================================
    # MÉTRICAS IoU / DICE
    # ======================================================

    mean_iou = (
        np.mean(ious)
        if len(ious)
        else 0
    )

    mean_dice = (
        np.mean(dices)
        if len(dices)
        else 0
    )

    # ======================================================
    # SALVAR RESULTADOS
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
print(
    df.mean(
        numeric_only=True
    )
)

print("\nDesvio padrão")
print(
    df.std(
        numeric_only=True
    )
)

# ==========================================================
# SALVAR RESULTADOS
# ==========================================================

results_path = os.path.join(
    BASE_DIR,
    "cross_validation_results.csv"
)

df.to_csv(
    results_path,
    index=False
)

print(
    f"\nResultados salvos em: {results_path}"
)
