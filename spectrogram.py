import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Caminho do arquivo MP3
audio_file = "audio-exemplo.mp3"

# Carregar áudio
y, sr = librosa.load(audio_file, sr=None)

# ==========================
# Espectrograma Linear (STFT)
# ==========================
D = np.abs(librosa.stft(y))
D_db = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(
    D_db,
    sr=sr,
    x_axis="time",
    y_axis="linear"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Espectrograma Linear")
plt.tight_layout()
plt.savefig("espectrograma_linear.png", dpi=300)
plt.close()

# ==========================
# Espectrograma Mel
# ==========================
S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)
S_db = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Espectrograma Mel")
plt.tight_layout()
plt.savefig("espectrograma_mel.png", dpi=300)
plt.close()

print("Arquivos salvos:")
print("- espectrograma_linear.png")
print("- espectrograma_mel.png")