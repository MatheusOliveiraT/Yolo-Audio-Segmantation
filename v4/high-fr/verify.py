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

AUDIO_PATH = os.getenv("AUDIO_PATH")
BASE_DIR = os.getenv("BASE_DIR")

PRED_DIR = os.path.join(BASE_DIR, "data/predictions/test/labels")
GT_DIR   = os.path.join(BASE_DIR, "data/labels/test")

OUT_DIR  = os.path.join(BASE_DIR, "reconstructions")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 32768
N_FFT = 2048
HOP = 1638
N_MELS = 128

WINDOW_SEC = 10
OVERLAP = 0.5
STEP = WINDOW_SEC * (1 - OVERLAP)

IOU_THRESHOLD = 0.3

# 🔥 ESCOLHA DO MODELO
MODE = "high"  # "low" | "high" | "full"

LOW_MAX = 3072
HIGH_MIN = 2048

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
# CONVERSÃO YOLO → TEMPO/FREQ
# =========================

def yolo_to_timefreq(box, t0, f_max):

    xc,yc,w,h = box

    ts = (xc - w/2) * WINDOW_SEC + t0
    te = (xc + w/2) * WINDOW_SEC + t0

    if MODE == "low":
        fs = (yc - h/2) * LOW_MAX
        fe = (yc + h/2) * LOW_MAX

    elif MODE == "high":
        fs = (yc - h/2) * (f_max - HIGH_MIN) + HIGH_MIN
        fe = (yc + h/2) * (f_max - HIGH_MIN) + HIGH_MIN

    else:  # full
        fs = (yc - h/2) * f_max
        fe = (yc + h/2) * f_max

    return ts,te,fs,fe

# =========================
# IoU
# =========================

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

# =========================
# LOAD AUDIO
# =========================

y,sr = librosa.load(AUDIO_PATH, sr=SR)
y = spectral_subtraction(y, sr)

S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS
)

S_db = librosa.power_to_db(S, ref=np.max)

duration = len(y)/sr
f_max = sr/2

# =========================
# MEL MASK (LOW / HIGH)
# =========================

mel_freqs = librosa.mel_frequencies(
    n_mels=N_MELS,
    fmin=0,
    fmax=f_max
)

if MODE == "low":
    mask = mel_freqs <= LOW_MAX
    S_db = S_db[mask,:]

elif MODE == "high":
    mask = mel_freqs >= HIGH_MIN
    S_db = S_db[mask,:]

# =========================
# PLOT
# =========================

plt.figure(figsize=(20,6))

librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=HOP,
    x_axis="time",
    y_axis="mel"
)

ax = plt.gca()

audio_id = os.path.basename(AUDIO_PATH).replace(".WAV","").replace(".wav","")

# =========================
# LOOP
# =========================

tiles = sorted(os.listdir(GT_DIR))

for file in tiles:

    if not file.startswith(audio_id):
        continue

    tile_id = int(file.split("_")[-1].replace(".txt",""))
    t0 = tile_id * STEP

    gt_boxes = []
    pred_boxes = []

    # ---------- GT ----------
    with open(os.path.join(GT_DIR,file)) as f:
        for line in f:
            cls,xc,yc,w,h = map(float,line.split())
            gt_boxes.append(yolo_to_timefreq((xc,yc,w,h), t0, f_max))

    # ---------- PRED ----------
    pred_path = os.path.join(PRED_DIR,file)

    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                cls,xc,yc,w,h,conf = map(float,line.split())
                pred_boxes.append(yolo_to_timefreq((xc,yc,w,h), t0, f_max))

    matched_gt = set()

    # =========================
    # MATCH
    # =========================

    for p in pred_boxes:

        best_iou = 0
        best_gi = None

        for gi,g in enumerate(gt_boxes):

            val = iou(p,g)

            if val > best_iou:
                best_iou = val
                best_gi = gi

        if best_iou >= IOU_THRESHOLD:

            matched_gt.add(best_gi)

            # TP
            ax.add_patch(plt.Rectangle(
                (p[0],p[2]),
                p[1]-p[0],
                p[3]-p[2],
                edgecolor="lime",
                fill=False,
                linewidth=2
            ))

        else:

            # FP
            ax.add_patch(plt.Rectangle(
                (p[0],p[2]),
                p[1]-p[0],
                p[3]-p[2],
                edgecolor="red",
                fill=False,
                linewidth=2
            ))

    # =========================
    # FN
    # =========================

    for gi,g in enumerate(gt_boxes):

        if gi not in matched_gt:

            ax.add_patch(plt.Rectangle(
                (g[0],g[2]),
                g[1]-g[0],
                g[3]-g[2],
                edgecolor="yellow",
                fill=False,
                linewidth=2
            ))

# =========================
# SAVE
# =========================

plt.title(f"{MODE.upper()} → TP=verde | FP=vermelho | FN=amarelo")

output_path = os.path.join(
    OUT_DIR,
    f"{audio_id}_{MODE}_error_map.png"
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Imagem salva em:", output_path)

plt.show()