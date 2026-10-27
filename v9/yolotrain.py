from ultralytics import YOLO
import os
import yaml

# =========================
# CAMINHOS
# =========================
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
DATA_DIR = os.path.join(BASE_DIR, "data3")

# =========================
# CRIAR YAML DO DATASET
# =========================
data_yaml = {
    "path": DATA_DIR,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["interest"]
}

yaml_path = os.path.join(DATA_DIR, "dataset.yaml")

with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print("📄 dataset.yaml criado")

# =========================
# CARREGAR MODELO
# =========================

model = YOLO("yolo26m.pt")  
# opções possíveis:
# yolov8n.pt  (mais rápido)
# yolov8s.pt  (melhor desempenho)
# yolov8m.pt

# =========================
# TREINO
# =========================

model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=64,  # ajuste conforme VRAM
    patience=20,
    workers=4,
    device=0,  # GPU (ou 'cpu')
    project=os.path.join(BASE_DIR, "runs"),
    name="yolo_audio_detection",
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
)
