from ultralytics import YOLO
import os
from dotenv import load_dotenv

def run_yolo_inference():
    load_dotenv()

    BASE_DIR = os.getenv("BASE_DIR")
    DATA_DIR = os.path.join(BASE_DIR, "data3")

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "runs3/yolo_audio_detection/weights/best.pt"
    )

    model = YOLO(MODEL_PATH)

    model.predict(
        source=os.path.join(DATA_DIR, "images/test"),
        imgsz=512,
        conf=0.25,
        save=False,
        save_txt=True,
        save_conf=True,
        project=os.path.join(DATA_DIR, "predictions"),
        name="test",
        exist_ok=True
    )

if __name__ == "__main__":
    run_yolo_inference()
