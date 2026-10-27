
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

OUT_DIR = os.path.join(BASE_DIR, "reconstructions")
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_DIR = "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104"

SR = 32768
N_FFT = 2048
HOP = 1638
N_MELS = 128

IOU_THRESHOLD = 0.3


# =========================
# ENCONTRAR ÁUDIOS DO TESTE
# =========================

def get_audio_ids_from_labels(label_dir):

    audio_ids = set()

    for file in os.listdir(label_dir):

        if not file.lower().endswith(".txt"):
            continue

        audio_id = os.path.splitext(file)[0]

        audio_ids.add(audio_id)

    return sorted(audio_ids)


audio_ids_test = get_audio_ids_from_labels(GT_DIR)


# =========================
# MAPEAR WAVs
# =========================

audio_paths = {}

for file in os.listdir(AUDIO_DIR):

    if not file.lower().endswith(".wav"):
        continue

    audio_id = os.path.splitext(file)[0]

    audio_paths[audio_id] = os.path.join(
        AUDIO_DIR,
        file
    )


# =========================
# FUNÇÕES
# =========================

def yolo_to_timefreq(box, duration, sr, n_mels):

    xc, yc, w, h = box

    # =========================
    # TEMPO
    # =========================

    ts = (xc - w / 2) * duration
    te = (xc + w / 2) * duration

    ts = max(0, ts)
    te = min(duration, te)

    # =========================
    # POSIÇÃO VERTICAL
    # =========================

    # YOLO:
    # yc = 0 -> topo
    # yc = 1 -> fundo
    #
    # Espectrograma:
    # frequência alta -> topo
    # frequência baixa -> fundo
    #
    # Portanto, invertemos:

    y_top = yc - h / 2
    y_bottom = yc + h / 2

    # Converte posição YOLO para posição
    # na escala mel, invertendo verticalmente

    mel_high_norm = 1 - y_top
    mel_low_norm = 1 - y_bottom

    # Limita [0, 1]

    mel_low_norm = np.clip(
        mel_low_norm,
        0,
        1
    )

    mel_high_norm = np.clip(
        mel_high_norm,
        0,
        1
    )

    # =========================
    # MEL -> HZ
    # =========================

    mel_min = librosa.hz_to_mel(0)

    mel_max = librosa.hz_to_mel(sr / 2)

    mel_low = (
        mel_min +
        mel_low_norm * (mel_max - mel_min)
    )

    mel_high = (
        mel_min +
        mel_high_norm * (mel_max - mel_min)
    )

    f_low = librosa.mel_to_hz(mel_low)
    f_high = librosa.mel_to_hz(mel_high)

    return ts, te, f_low, f_high

def intersection(a, b):

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])

    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    return (
        max(0, xB - xA) *
        max(0, yB - yA)
    )


def iou(a, b):

    inter = intersection(a, b)

    areaA = (
        (a[2] - a[0]) *
        (a[3] - a[1])
    )

    areaB = (
        (b[2] - b[0]) *
        (b[3] - b[1])
    )

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


# =========================
# PROCESSAR UM ÁUDIO
# =========================

def process_audio(audio_id, audio_path):

    tp = 0
    fp = 0
    fn = 0

    # =========================
    # CARREGAR ÁUDIO
    # =========================

    y, sr = librosa.load(
        audio_path,
        sr=SR
    )

    duration = len(y) / sr

    # =========================
    # MEL SPECTROGRAM
    # =========================

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=0,
        fmax=sr / 2
    )

    S_db = librosa.power_to_db(
        S,
        ref=np.max
    )

    # =========================
    # PLOT
    # =========================

    plt.figure(figsize=(20, 6))

    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP,
        x_axis="time",
        y_axis="mel",
        fmin=0,
        fmax=sr / 2
    )

    ax = plt.gca()

    # =========================
    # GT
    # =========================

    gt_path = os.path.join(
        GT_DIR,
        f"{audio_id}.txt"
    )

    gt_boxes = []

    if os.path.exists(gt_path):

        with open(gt_path) as f:

            for line in f:

                values = list(
                    map(float, line.split())
                )

                if len(values) < 5:
                    continue

                cls, xc, yc, w, h = values[:5]

                ts, te, fs, fe = yolo_to_timefreq(
                    (xc, yc, w, h),
                    duration,
                    sr,
                    N_MELS
                )

                gt_boxes.append(
                    [ts, fs, te, fe]
                )

    else:

        print(
            f"⚠️ GT não encontrado: {gt_path}"
        )

    # =========================
    # PRED
    # =========================

    pred_path = os.path.join(
        PRED_DIR,
        f"{audio_id}.txt"
    )

    pred_boxes = []

    if os.path.exists(pred_path):

        with open(pred_path) as f:

            for line in f:

                values = list(
                    map(float, line.split())
                )

                if len(values) < 5:
                    continue

                # YOLO prediction normalmente:
                #
                # cls xc yc w h conf

                cls, xc, yc, w, h = values[:5]

                ts, te, fs, fe = yolo_to_timefreq(
                    (xc, yc, w, h),
                    duration,
                    sr,
                    N_MELS
                )

                pred_boxes.append(
                    [ts, fs, te, fe]
                )

    # =========================
    # MATCHING
    # =========================

    matched_gt = set()
    matched_pred = set()

    for pi, p in enumerate(pred_boxes):

        best_iou = 0
        best_gi = None

        for gi, g in enumerate(gt_boxes):

            # Não permite que o mesmo GT
            # seja utilizado por duas predições

            if gi in matched_gt:
                continue

            val = iou(p, g)

            if val > best_iou:

                best_iou = val
                best_gi = gi

        # -------------------------
        # TP
        # -------------------------

        if best_iou >= IOU_THRESHOLD:

            matched_gt.add(best_gi)
            matched_pred.add(pi)

            tp += 1

            ax.add_patch(
                plt.Rectangle(
                    (p[0], p[1]),
                    p[2] - p[0],
                    p[3] - p[1],
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=2
                )
            )

        # -------------------------
        # FP
        # -------------------------

        else:

            fp += 1

            ax.add_patch(
                plt.Rectangle(
                    (p[0], p[1]),
                    p[2] - p[0],
                    p[3] - p[1],
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2
                )
            )

    # =========================
    # FN
    # =========================

    for gi, g in enumerate(gt_boxes):

        if gi not in matched_gt:

            fn += 1

            ax.add_patch(
                plt.Rectangle(
                    (g[0], g[1]),
                    g[2] - g[0],
                    g[3] - g[1],
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=2
                )
            )

    # =========================
    # TÍTULO
    # =========================

    plt.title(
        f"{audio_id} | "
        f"TP={tp} | FP={fp} | FN={fn}"
    )

    # =========================
    # SALVAR
    # =========================

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

    plt.close()

    # =========================
    # MÉTRICAS
    # =========================

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    print(
        f"✅ Processado: {audio_id} | "
        f"TP={tp} FP={fp} FN={fn} | "
        f"F1={f1:.4f}"
    )

    return {
        "audio": audio_path,
        "audio_id": audio_id,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================
# PROCESSAR TODOS OS ÁUDIOS
# =========================

results = []

for audio_id in audio_ids_test:

    if audio_id not in audio_paths:

        print(
            f"⚠️ WAV não encontrado: {audio_id}"
        )

        continue

    audio_path = audio_paths[audio_id]

    try:

        result = process_audio(
            audio_id,
            audio_path
        )

        results.append(result)

    except Exception as e:

        print(
            f"❌ Erro em {audio_id}: {e}"
        )


# =========================
# DATAFRAME
# =========================

results_df = pd.DataFrame(results)

if results_df.empty:

    print(
        "\n❌ Nenhum áudio foi processado."
    )

    exit()


# =========================
# TOP 5
# =========================

top5 = results_df.sort_values(
    "f1",
    ascending=False
).head(5)

print(
    "\n===== 5 MELHORES ÁUDIOS ====="
)

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


# =========================
# BOTTOM 5
# =========================

bottom5 = results_df.sort_values(
    "f1",
    ascending=True
).head(5)

print(
    "\n===== 5 PIORES ÁUDIOS ====="
)

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


# =========================
# 5 MAIS PRÓXIMOS DA MEDIANA
# =========================

median_f1 = results_df["f1"].median()

results_df["distance_to_median"] = (
    results_df["f1"] - median_f1
).abs()

middle5 = results_df.sort_values(
    "distance_to_median"
).head(5)

print(
    "\n===== 5 ÁUDIOS MÉDIOS ====="
)

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


# =========================
# CSV
# =========================

results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "test_audio_metrics.csv"
    ),
    index=False
)


# =========================
# RANKING COMPLETO
# =========================

ranking = results_df.sort_values(
    "f1",
    ascending=False
)

print(
    "\n===== RANKING COMPLETO ====="
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
