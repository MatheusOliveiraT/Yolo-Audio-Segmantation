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

AUDIO_LIST = ["/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04821899S2014959_20231104_084000.WAV", 
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04822980S2014203_20231110_062000.WAV", 
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04822980S2014203_20240307_071000.WAV",
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04831025S2014237_20231220_071000.WAV",
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04831109S2013814_20240514_070000.WAV",
              "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104/W04857032S2013387_20231225_052000.WAV"]

BASE_DIR = os.getenv("BASE_DIR")

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

    y,sr = librosa.load(AUDIO_PATH, sr=SR)

    y = spectral_subtraction(y, sr)
    y = spectral_difference(y, sr)

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

    print(f"✅ Processado: {audio_id}")

    plt.show()

for audio_path in AUDIO_LIST:
    process_audio(audio_path)