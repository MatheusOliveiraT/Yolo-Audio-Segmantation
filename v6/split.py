import os
import glob
import random
import shutil
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")

IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

DATA_DIR = os.path.join(BASE_DIR, "data2")

# =========================
# listar imagens
# =========================

images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))

# agrupar por áudio
groups = defaultdict(list)

for img in images:
    name = os.path.basename(img)

    # remove o índice do tile
    audio_id = "_".join(name.split("_")[:-1])

    groups[audio_id].append(img)

audio_ids = list(groups.keys())

print("Total de áudios:", len(audio_ids))

# =========================
# split
# =========================

random.seed(24)
random.shuffle(audio_ids)

train_split = 0.7
val_split = 0.2

n = len(audio_ids)

train_ids = audio_ids[:int(n * train_split)]
val_ids = audio_ids[int(n * train_split):int(n * (train_split + val_split))]
test_ids = audio_ids[int(n * (train_split + val_split)):]

print("train:", len(train_ids))
print("val:", len(val_ids))
print("test:", len(test_ids))

# =========================
# função para copiar
# =========================

def copy_group(ids, split):

    img_out = os.path.join(DATA_DIR, "images", split)
    lbl_out = os.path.join(DATA_DIR, "labels", split)

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for aid in ids:

        for img_path in groups[aid]:

            name = os.path.basename(img_path)
            label = name.replace(".jpg", ".txt")

            shutil.copy(img_path, os.path.join(img_out, name))

            lbl_path = os.path.join(LBL_DIR, label)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(lbl_out, label))


copy_group(train_ids, "train")
copy_group(val_ids, "val")
copy_group(test_ids, "test")

print("✅ Dataset dividido corretamente por áudio")
