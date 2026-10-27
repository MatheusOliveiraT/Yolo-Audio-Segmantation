import resource

soft, hard = resource.getrlimit(resource.RLIMIT_AS)

print(f"Limite atual: soft={soft}, hard={hard}")

MEM_GB = 8
mem_bytes = MEM_GB * 1024**3

resource.setrlimit(
    resource.RLIMIT_AS,
    (min(mem_bytes, hard), hard)
)

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from dotenv import load_dotenv

# =========================
# CONFIGURAÇÕES
# =========================

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH")
OUT_BASE = os.getenv("BASE_DIR")

IMG_DIR = os.path.join(OUT_BASE, "images")
LBL_DIR = os.path.join(OUT_BASE, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

SR = 2**15
N_FFT = 2048
HOP = 1638
N_MELS = 128

HIGH_MIN = 2048  # 🔥 início da banda HIGH

WINDOW_SEC = 10.0
OVERLAP = 0.5

# =========================
# CSV
# =========================

df = pd.read_csv(CSV_PATH)
grouped = df.groupby("soundscape_path")

print(f"🎧 Total de áudios no CSV: {len(grouped)}")

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
# PROCESSAMENTO
# =========================

for audio_path, roi_df in tqdm(grouped, desc="HIGH model"):

    if not os.path.exists(audio_path):
        continue

    try:
        # -------------------------
        # Load + denoise
        # -------------------------
        y, sr = librosa.load(audio_path, sr=SR)
        y = spectral_subtraction(y, sr)

        duration = len(y) / sr
        f_max = sr / 2

        # -------------------------
        # Mel spectrogram
        # -------------------------
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP,
            n_mels=N_MELS,
            fmax=f_max
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 🔥 MEL FREQUENCIES
        mel_freqs = librosa.mel_frequencies(
            n_mels=N_MELS,
            fmin=0,
            fmax=f_max
        )

        # 🔥 HIGH MASK
        high_mask = mel_freqs >= HIGH_MIN

        S_high = S_dB[high_mask, :]

        # -------------------------
        # Tiling
        # -------------------------
        step = WINDOW_SEC * (1 - OVERLAP)
        times = np.arange(0, duration - WINDOW_SEC, step)

        base = os.path.splitext(os.path.basename(audio_path))[0]

        for i, t0 in enumerate(times):

            t1 = t0 + WINDOW_SEC

            f_start = int(t0 * sr / HOP)
            f_end = int(t1 * sr / HOP)

            S_slice = S_high[:, f_start:f_end]

            if S_slice.shape[1] < 5:
                continue

            # -------------------------
            # Salvar imagem
            # -------------------------
            img_name = f"{base}_{i:05d}.jpg"
            img_path = os.path.join(IMG_DIR, img_name)

            plt.figure(figsize=(5, 5))
            plt.imshow(S_slice, aspect="auto", origin="lower", cmap="magma")
            plt.axis("off")
            plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            # -------------------------
            # LABEL YOLO (HIGH)
            # -------------------------
            label_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

            with open(label_path, "w") as f:

                for _, row in roi_df.iterrows():

                    rs, re = row["roi_start"], row["roi_end"]

                    if re <= t0 or rs >= t1:
                        continue

                    ns = max(rs, t0) - t0
                    ne = min(re, t1) - t0

                    if ne <= ns:
                        continue

                    fmin = row["roi_min_freq"] * 1000
                    fmax = row["roi_max_freq"] * 1000

                    # 🔥 CLIP PARA HIGH
                    fmin_c = max(HIGH_MIN, fmin)
                    fmax_c = min(f_max, fmax)

                    if fmax_c <= fmin_c:
                        continue

                    # -------------------------
                    # NORMALIZAÇÃO COM OFFSET
                    # -------------------------
                    xc = (ns + ne) / 2 / WINDOW_SEC
                    w = (ne - ns) / WINDOW_SEC

                    yc = ((fmin_c + fmax_c) / 2 - HIGH_MIN) / (f_max - HIGH_MIN)
                    h  = (fmax_c - fmin_c) / (f_max - HIGH_MIN)

                    xc, yc = np.clip(xc, 0, 1), np.clip(yc, 0, 1)
                    w, h = np.clip(w, 0, 1), np.clip(h, 0, 1)

                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        del y, S, S_dB, S_high
        gc.collect()

    except Exception as e:
        print(f"Erro em {audio_path}: {e}")

print("✅ Dataset HIGH pronto!")