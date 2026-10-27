import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from dotenv import load_dotenv

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH")
OUT_BASE = os.getenv("BASE_DIR")

IMG_DIR = os.path.join(OUT_BASE, "images")
LBL_DIR = os.path.join(OUT_BASE, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# Áudio / espectrograma
SR = 2**15            # 32768 Hz
N_FFT = 2048
HOP = 1638
N_MELS = 128

# Tiling temporal
WINDOW_SEC = 10.0
OVERLAP = 0.5

# =========================
# LER CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Usamos soundscape_path como referência do WAV
grouped = df.groupby("soundscape_path")

print(f"🎧 Total de áudios no CSV: {len(grouped)}")

# =========================
# SPECTRAL SUBTRACTION
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

# =========================
# PROCESSAR CADA WAV
# =========================
for audio_path, roi_df in tqdm(grouped, desc="Processando WAVs"):
    if not os.path.exists(audio_path):
        print(f"⚠️ Arquivo não encontrado: {audio_path}")
        continue

    try:
        # -------------------------
        # Carregar áudio
        # -------------------------
        y, sr = librosa.load(audio_path, sr=SR)

        # =========================
        # PRÉ-PROCESSAMENTO
        # =========================

        # 1) spectral subtraction (leve)
        # y = spectral_subtraction(y, sr)

        # 2) diferença espectral
        y = spectral_difference(y, sr)

        duration = len(y) / sr
        f_max = sr / 2

        # -------------------------
        # Mel spectrograma
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

        # -------------------------
        # Tiling temporal
        # -------------------------
        step = WINDOW_SEC * (1 - OVERLAP)
        times = np.arange(0, duration - WINDOW_SEC, step)

        base = os.path.splitext(os.path.basename(audio_path))[0]

        for i, t0 in enumerate(times):
            t1 = t0 + WINDOW_SEC

            f_start = int(t0 * sr / HOP)
            f_end = int(t1 * sr / HOP)

            S_slice = S_dB[:, f_start:f_end]
            if S_slice.shape[1] < 5:
                continue

            # -------------------------
            # Salvar imagem
            # -------------------------
            img_name = f"{base}_{i:05d}.jpg"
            img_path = os.path.join(IMG_DIR, img_name)

            plt.figure(figsize=(5, 5))
            librosa.display.specshow(
                S_slice,
                sr=sr,
                hop_length=HOP,
                y_axis="mel",
                x_axis=None,
                fmax=f_max,
                cmap="magma"
            )
            plt.axis("off")
            plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            # -------------------------
            # Gerar label YOLO
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

                    xc = (ns + ne) / 2 / WINDOW_SEC
                    w = (ne - ns) / WINDOW_SEC
                    yc = (fmin + fmax) / 2 / f_max
                    h = (fmax - fmin) / f_max

                    # Clamp
                    xc, yc = np.clip(xc, 0, 1), np.clip(yc, 0, 1)
                    w, h = np.clip(w, 0, 1), np.clip(h, 0, 1)

                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        del y, S, S_dB
        gc.collect()

    except Exception as e:
        print(f"❌ Erro ao processar {audio_path}: {e}")

print("✅ Dataset YOLO gerado para todos os WAVs!")