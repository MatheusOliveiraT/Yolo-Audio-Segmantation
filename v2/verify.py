import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

AUDIO_PATH = "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04857032S2013387_20231225_052000.WAV"

BASE_DIR = "/home/a_matheusteixeira/yolo_dataset"

PRED_DIR = os.path.join(BASE_DIR, "data/predictions/test/labels")
GT_DIR   = os.path.join(BASE_DIR, "data/labels/test")

OUT_DIR  = os.path.join(BASE_DIR, "reconstructions")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 32768
N_FFT = 2048
HOP = 1638

WINDOW_SEC = 10
OVERLAP = 0.5
STEP = WINDOW_SEC * (1 - OVERLAP)

IOU_THRESHOLD = 0.3

# =========================
# FUNÇÕES
# =========================

def yolo_to_timefreq(box, t0, f_max):

    xc,yc,w,h = box

    ts = (xc - w/2) * WINDOW_SEC + t0
    te = (xc + w/2) * WINDOW_SEC + t0

    fs = (yc - h/2) * f_max
    fe = (yc + h/2) * f_max

    return ts,te,fs,fe


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
# CARREGAR AUDIO
# =========================

y,sr = librosa.load(AUDIO_PATH, sr=SR)

S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP
)

S_db = librosa.power_to_db(S, ref=np.max)

duration = len(y)/sr
f_max = sr/2

audio_id = os.path.basename(AUDIO_PATH).replace(".WAV","").replace(".wav","")

# =========================
# PLOT BASE
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

# =========================
# PERCORRER TILES
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

            ts,te,fs,fe = yolo_to_timefreq((xc,yc,w,h), t0, f_max)

            gt_boxes.append([ts,fs,te,fe])


    # ---------- PRED ----------
    pred_path = os.path.join(PRED_DIR,file)

    if os.path.exists(pred_path):

        with open(pred_path) as f:

            for line in f:

                cls,xc,yc,w,h,conf = map(float,line.split())

                ts,te,fs,fe = yolo_to_timefreq((xc,yc,w,h), t0, f_max)

                pred_boxes.append([ts,fs,te,fe])


    matched_gt = set()
    matched_pred = set()

    # =========================
    # MATCHING
    # =========================

    for pi,p in enumerate(pred_boxes):

        best_iou = 0
        best_gt = None
        best_gi = None

        for gi,g in enumerate(gt_boxes):

            val = iou(p,g)

            if val > best_iou:

                best_iou = val
                best_gt = g
                best_gi = gi

        if best_iou >= IOU_THRESHOLD:

            matched_gt.add(best_gi)
            matched_pred.add(pi)

            # TP
            ax.add_patch(
                plt.Rectangle(
                    (p[0],p[1]),
                    p[2]-p[0],
                    p[3]-p[1],
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=2
                )
            )

        else:

            # FP
            ax.add_patch(
                plt.Rectangle(
                    (p[0],p[1]),
                    p[2]-p[0],
                    p[3]-p[1],
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2
                )
            )


    # =========================
    # FN
    # =========================

    for gi,g in enumerate(gt_boxes):

        if gi not in matched_gt:

            ax.add_patch(
                plt.Rectangle(
                    (g[0],g[1]),
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

plt.title("TP=verde | FP=vermelho | FN=amarelo")

output_path = os.path.join(
    OUT_DIR,
    f"{audio_id}_error_map.png"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("Imagem salva em:", output_path)

plt.show()
