import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =====================================================
# CONFIGURAÇÕES
# =====================================================

CSV_PATH = "/home/a_matheusteixeira/df_ROI_RosaGLM_ConservaSom_20241104_certeza.csv"

AUDIO_COLUMN = "soundscape_path"

SPECIES = "Patagioenas picazuro"

OUTPUT_DIR = "/home/a_matheusteixeira/patagioenas_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# CARREGA DADOS
# =====================================================

df = pd.read_csv(CSV_PATH)

# ajuste o nome da coluna caso seja diferente
species_col = "roi_label"

df = df[df[species_col] == SPECIES].copy()

if len(df) == 0:
    raise ValueError(f"Nenhuma ROI encontrada para {SPECIES}")

# =====================================================
# TAMANHO DA ROI (EM SEGUNDOS)
# =====================================================

df["roi_duration"] = df["roi_end"] - df["roi_start"]

p10 = np.percentile(df["roi_duration"], 10)
p90 = np.percentile(df["roi_duration"], 90)

roi_p10 = df.iloc[
    (df["roi_duration"] - p10).abs().argmin()
]

roi_p90 = df.iloc[
    (df["roi_duration"] - p90).abs().argmin()
]

print("ROI P10")
print(roi_p10[[
    "soundscape_file",
    "roi_start",
    "roi_end",
    "roi_min_freq",
    "roi_max_freq",
    "roi_duration",
    "roi_user"
]])

print("\nROI P90")
print(roi_p90[[
    "soundscape_file",
    "roi_start",
    "roi_end",
    "roi_min_freq",
    "roi_max_freq",
    "roi_duration",
    "roi_user"
]])

# =====================================================
# FUNÇÃO PARA GERAR ESPECTROGRAMA
# =====================================================

def save_mel_roi(row, label):
    audio_path = row[AUDIO_COLUMN]

    roi_start = float(row["roi_start"])
    roi_end = float(row["roi_end"])

    roi_min_freq_hz = float(row["roi_min_freq"]) * 1000
    roi_max_freq_hz = float(row["roi_max_freq"]) * 1000

    duration = roi_end - roi_start

    # margem para visualizar contexto
    margin = max(1.0, duration)

    plot_start = max(0, roi_start - margin)
    plot_end = roi_end + margin

    print(f"\nCarregando: {audio_path}")

    y, sr = librosa.load(
        audio_path,
        sr=None,
        offset=plot_start,
        duration=(plot_end - plot_start)
    )

    # espectrograma Mel
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=1638,
        n_mels=128,
        fmax=sr // 2
    )

    mel_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=1638,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    # plt.colorbar(img, ax=ax, format="%+2.f dB")

    # posição relativa da ROI no gráfico
    roi_x = roi_start - plot_start

    rect = Rectangle(
        (roi_x, roi_min_freq_hz),
        duration,
        roi_max_freq_hz - roi_min_freq_hz,
        fill=False,
        linewidth=2,
        edgecolor="lime"
    )

    ax.add_patch(rect)

    ax.set_title(
        f"{SPECIES}\n"
        f"{label} | duração={duration:.3f}s"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{label}_mel_spectrogram.png"
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Salvo em: {output_file}")

# =====================================================
# GERA FIGURAS
# =====================================================

save_mel_roi(roi_p10, "P10")
save_mel_roi(roi_p90, "P90")