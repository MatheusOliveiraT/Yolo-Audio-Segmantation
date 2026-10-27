
from ultralytics import YOLO
import os
import yaml

BASE_DIR = "/home/a_matheusteixeira/yolo_dataset_old_cv"
DATA_DIR = os.path.join(BASE_DIR, "data")

yaml_path = os.path.join(DATA_DIR, "dataset_final.yaml")

data_yaml = {
    "path": "",                  # caminhos já são absolutos
    "train": train_txt,
    "val": train_txt,            
    "test": test_txt,
    "nc": 1,
    "names": ["interest"]
}

with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"YAML salvo em: {yaml_path}")

model = YOLO("yolo26m.pt")

model.train(
    data=yaml_path,
    epochs=200,
    imgsz=640,
    batch=128,
    device=0,
    workers=4,
    optimizer="AdamW",
    pretrained=True,
    val=False,
    project=os.path.join(BASE_DIR, "runs"),
    name="final_model"
)

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

print(results)
