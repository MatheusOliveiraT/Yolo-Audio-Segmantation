import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")

PRED_DIR = os.path.join(BASE_DIR, "data/predictions/test/labels")
GT_DIR   = os.path.join(BASE_DIR, "data/labels/test")

OUT_DIR  = os.path.join(BASE_DIR, "reconstructions")
os.makedirs(OUT_DIR, exist_ok=True)

def get_audio_ids_from_labels(label_dir):
    audio_ids = set()
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        name = file.replace(".txt", "")
        # remove o índice do tile
        audio_id = "_".join(name.split("_")[:-1])
        audio_ids.add(audio_id)
    return sorted(audio_ids)

audio_ids_test = get_audio_ids_from_labels(GT_DIR)

AUDIO_DIR = "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104"

audio_paths = {}

for file in os.listdir(AUDIO_DIR):
    if not file.lower().endswith(".wav"):
        continue
    audio_id = os.path.splitext(file)[0]
    audio_paths[audio_id] = os.path.join(
        AUDIO_DIR,
        file
    )

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

def spectral_subtraction(y, sr):

    n_fft = 2048
    hop = 512

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)

    # estimar ruído (primeiros 0.5s)
    noise_frames = int(0.5 * sr / hop)
    noise_profile = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

    # subtração leve
    alpha = 1.0  # intensidade (0.5–2.0)
    mag_denoised = mag - alpha * noise_profile

    # floor para não zerar sinal útil
    mag_denoised = np.maximum(mag_denoised, 0.1 * mag)

    # reconstrução
    D_denoised = mag_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(D_denoised, hop_length=hop)

    return y_denoised

def spectral_difference(y, sr):

    n_fft = 2048
    hop = 512

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)

    # =========================
    # DIFERENÇA ESPECTRAL (temporal)
    # =========================

    # diferença entre frames consecutivos
    diff_mag = np.diff(mag, axis=1)

    # padding para manter dimensões
    diff_mag = np.pad(diff_mag, ((0, 0), (1, 0)), mode='constant')

    # manter apenas variações positivas (onsets / mudanças)
    diff_mag = np.maximum(diff_mag, 0)

    # normalização opcional
    diff_mag = diff_mag / (np.max(diff_mag) + 1e-8)

    # =========================
    # RECONSTRUÇÃO
    # =========================

    D_diff = diff_mag * np.exp(1j * phase)
    y_diff = librosa.istft(D_diff, hop_length=hop)

    return y_diff

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

def process_audio(AUDIO_PATH):

    tp = 0
    fp = 0
    fn = 0

    y,sr = librosa.load(AUDIO_PATH, sr=SR)

    y = spectral_subtraction(y, sr)
    #y = spectral_difference(y, sr)

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
                tp += 1

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
                fp += 1

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

                fn += 1

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

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    print(f"✅ Processado: {audio_id}")

    return {
        "audio": AUDIO_PATH,
        "audio_id": audio_id,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

results = []

for audio_id in audio_ids_test:
    if audio_id not in audio_paths:
        print(f"⚠️ WAV não encontrado: {audio_id}")
        continue
    audio_path = audio_paths[audio_id]
    try:
        result = process_audio(audio_path)
        results.append(result)
    except Exception as e:
        print(f"❌ Erro em {audio_id}: {e}")

results_df = pd.DataFrame(results)
print(results_df)

top5 = results_df.sort_values(
    "f1",
    ascending=False
).head(5)

print("\n===== 5 MELHORES ÁUDIOS =====")
print(
    top5[
        [
            "audio_id",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)

bottom5 = results_df.sort_values(
    "f1",
    ascending=True
).head(5)

print("\n===== 5 PIORES ÁUDIOS =====")
print(
    bottom5[
        [
            "audio_id",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)

median_f1 = results_df["f1"].median()
results_df["distance_to_median"] = (
    results_df["f1"] - median_f1
).abs()
middle5 = results_df.sort_values(
    "distance_to_median"
).head(5)

print("\n===== 5 ÁUDIOS MÉDIOS =====")
print(
    middle5[
        [
            "audio_id",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)

results_df.to_csv(
    os.path.join(BASE_DIR, "test_audio_metrics.csv"),
    index=False
)

ranking = results_df.sort_values(
    "f1",
    ascending=False
)

print(
    ranking[
        [
            "audio_id",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)
