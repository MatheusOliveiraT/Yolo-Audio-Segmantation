import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================

load_dotenv()

BASE_HIGH = os.getenv("BASE_HIGH")
BASE_LOW = os.getenv("BASE_LOW")

PRED_LOW_DIR = os.path.join(BASE_LOW, "data/predictions/test/labels")
PRED_HIGH_DIR = os.path.join(BASE_HIGH, "data/predictions/test/labels")
GT_LOW_DIR = os.path.join(BASE_LOW, "data/labels/test")
GT_HIGH_DIR = os.path.join(BASE_HIGH, "data/labels/test")

OUT_DIR  = os.path.join(BASE_LOW, "reconstructions")
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_LIST = ["/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04821899S2014959_20231104_084000.WAV", 
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04822980S2014203_20231110_062000.WAV", 
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04822980S2014203_20240307_071000.WAV",
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04831025S2014237_20231220_071000.WAV",
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04831109S2013814_20240514_070000.WAV",
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04857032S2013387_20231225_052000.WAV"]

SR = 32768
N_FFT = 2048
HOP = 1638

WINDOW_SEC = 10
OVERLAP = 0.5
STEP = WINDOW_SEC * (1 - OVERLAP)

IOU_THRESHOLD = 0.3

# =========================
# SPECTRAL SUBTRACTION
# =========================

def spectral_subtraction(y, sr):

    D = librosa.stft(y, n_fft=2048, hop_length=512)
    mag, phase = np.abs(D), np.angle(D)

    noise_frames = int(0.5 * sr / 512)
    noise_profile = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    mag_denoised = mag - noise_profile
    mag_denoised = np.maximum(mag_denoised, 0.1 * mag)

    D_denoised = mag_denoised * np.exp(1j * phase)
    return librosa.istft(D_denoised, hop_length=512)


# =========================
# FUNÇÕES
# =========================

LOW_MAX = 3072
HIGH_MIN = 2048

def yolo_to_timefreq(box, t0, f_max, band):

    xc, yc, w, h = box

    # tempo (igual)
    ts = (xc - w/2) * WINDOW_SEC + t0
    te = (xc + w/2) * WINDOW_SEC + t0

    # frequência (CORRIGIDO)
    if band == "low":

        fs = (yc - h/2) * LOW_MAX
        fe = (yc + h/2) * LOW_MAX

    elif band == "high":

        fs = (yc - h/2) * (f_max - HIGH_MIN) + HIGH_MIN
        fe = (yc + h/2) * (f_max - HIGH_MIN) + HIGH_MIN

    else:
        raise ValueError("band must be 'low' or 'high'")

    return ts, te, fs, fe


def intersection(a,b):

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    return max(0, xB-xA) * max(0, yB-yA)


def iou(a,b):

    inter = intersection(a,b)

    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])

    union = areaA + areaB - inter

    return inter/union if union>0 else 0

def load_boxes(label_path, t0, f_max, band, is_pred=False):

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path) as f:
        for line in f:

            parts = list(map(float, line.split()))

            if is_pred:
                cls, xc, yc, w, h, conf = parts
            else:
                cls, xc, yc, w, h = parts

            ts, te, fs, fe = yolo_to_timefreq(
                (xc, yc, w, h), t0, f_max, band
            )

            boxes.append([ts, fs, te, fe])

    return boxes


def process_audio(AUDIO_PATH):

    audio_id = os.path.basename(AUDIO_PATH).replace(".WAV","").replace(".wav","")

    y, sr = librosa.load(AUDIO_PATH, sr=SR)

    y = spectral_subtraction(y, sr)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    duration = len(y)/sr
    f_max = sr/2

    plt.figure(figsize=(20,6))

    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP,
        x_axis="time",
        y_axis="mel"
    )

    ax = plt.gca()

    # =========================
    # TILES (LOW + HIGH)
    # =========================

    tiles_low  = {f for f in os.listdir(GT_LOW_DIR) if f.endswith(".txt")}
    tiles_high = {f for f in os.listdir(GT_HIGH_DIR) if f.endswith(".txt")}

    tiles = sorted(tiles_low.union(tiles_high))

    for file in tiles:

        if not file.startswith(audio_id):
            continue

        tile_id = int(file.split("_")[-1].replace(".txt",""))
        t0 = tile_id * STEP

        gt_boxes = []
        pred_boxes = []

        # LOW
        gt_boxes += load_boxes(
            os.path.join(GT_LOW_DIR, file),
            t0, f_max, "low"
        )

        pred_boxes += load_boxes(
            os.path.join(PRED_LOW_DIR, file),
            t0, f_max, "low", is_pred=True
        )

        # HIGH
        gt_boxes += load_boxes(
            os.path.join(GT_HIGH_DIR, file),
            t0, f_max, "high"
        )

        pred_boxes += load_boxes(
            os.path.join(PRED_HIGH_DIR, file),
            t0, f_max, "high", is_pred=True
        )

        # =========================
        # MATCHING (igual ao seu)
        # =========================

        matched_gt = set()

        for p in pred_boxes:

            best_iou = 0
            best_gi = None

            for gi, g in enumerate(gt_boxes):

                val = iou(p, g)

                if val > best_iou:
                    best_iou = val
                    best_gi = gi

            if best_iou >= IOU_THRESHOLD:

                matched_gt.add(best_gi)

                ax.add_patch(
                    plt.Rectangle(
                        (p[0], p[1]),
                        p[2]-p[0],
                        p[3]-p[1],
                        edgecolor="lime",
                        facecolor="none",
                        linewidth=2
                    )
                )

            else:
                ax.add_patch(
                    plt.Rectangle(
                        (p[0], p[1]),
                        p[2]-p[0],
                        p[3]-p[1],
                        edgecolor="red",
                        facecolor="none",
                        linewidth=2
                    )
                )

        # FN
        for gi, g in enumerate(gt_boxes):

            if gi not in matched_gt:

                ax.add_patch(
                    plt.Rectangle(
                        (g[0], g[1]),
                        g[2]-g[0],
                        g[3]-g[1],
                        edgecolor="yellow",
                        facecolor="none",
                        linewidth=2
                    )
                )

    # =========================
    # SALVAR
    # =========================

    output_path = os.path.join(
        OUT_DIR,
        f"{audio_id}_error_map.png"
    )

    plt.title("TP=verde | FP=vermelho | FN=amarelo")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Processado: {audio_id}")

for audio_path in AUDIO_LIST:
    process_audio(audio_path)