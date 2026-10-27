
import os
import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # barra de progresso
from scipy.signal import butter, lfilter
from dotenv import load_dotenv

# === CONFIGURAÇÕES ===

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")

csv_path = os.getenv("CSV_SAMPLED_PATH")
audio_dir = os.getenv("AUDIO_DIR")
output_img_dir = os.path.join(BASE_DIR, "images")
output_label_dir = os.path.join(BASE_DIR, "labels")

# Parâmetros do espectrograma
sr = 2**15
n_fft = 2048
hop_length = 1638
n_mels = 128       # mais resolução vertical
img_size = (640, 640)

# Criar pastas de saída
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# === LER CSV ===
df = pd.read_csv(csv_path)

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

# === Iterar por arquivo de áudio ===
for audio_path, roi_df in tqdm(df.groupby("soundscape_path"), desc="Gerando dataset"):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_txt = os.path.join(output_label_dir, f"{base_name}.txt")
    output_img = os.path.join(output_img_dir, f"{base_name}.jpg")

    # === Carregar áudio ===
    try:
        y, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"[ERRO] {audio_path}: {e}")
        continue

    y = spectral_subtraction(y, sr)
    duration = len(y) / sr
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)

    # === Gerar espectrograma Mel ===
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr/2
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # === Normalizar para 0–255 e converter em imagem ===
    img = (255 * (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
    img = cv2.flip(img, 0)  # inverter verticalmente para alinhar frequências

    # Ajustar proporção (para 640x640 mantendo altura mais perceptiva)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    # === Converter ROIs para formato YOLO ===
    h, w, _ = img.shape
    lines = []

    for _, row in roi_df.iterrows():
        start = row["roi_start"]
        end = row["roi_end"]
        fmin = row["roi_min_freq"] * 1000
        fmax = row["roi_max_freq"] * 1000

        # Converter Hz → índice Mel
        fmin_idx = np.interp(fmin, mel_freqs, np.arange(len(mel_freqs)))
        fmax_idx = np.interp(fmax, mel_freqs, np.arange(len(mel_freqs)))

        # Coordenadas normalizadas (0–1)
        x_center = ((start + end) / 2) / duration
        y_center = 1 - ((fmin_idx + fmax_idx) / 2 / n_mels)  # flip eixo Y
        width = (end - start) / duration
        height = (fmax_idx - fmin_idx) / n_mels

        # Clamps
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        class_id = 0
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # === Salvar labels YOLO ===
    with open(output_txt, "w") as f:
        f.write("\n".join(lines))

    # === Salvar imagem ===
    cv2.imwrite(output_img, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

print("\n✅ Dataset YOLO gerado com sucesso!")

