
from ultralytics import YOLO
import os
import yaml

BASE_DIR = "/home/a_matheusteixeira/yolo_dataset_old_cv_20"
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs/fold_5/weights/best.pt"
)

model = YOLO(MODEL_PATH)

# === Gerar predições para TEST ===
results = model.predict(
    source=os.path.join(DATA_DIR, "images/test"),
    imgsz=640,
    conf=0.25,
    save=False,
    save_txt=True,        # 🔥 caixas YOLO em TXT
    save_conf=True,       # salva confiança
    project=os.path.join(DATA_DIR, "predictions"),
    name="test"
)
