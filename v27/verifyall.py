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

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

GT_DIR = os.path.join(
    DATA_DIR,
    "labels",
    "test"
)

PRED_BASE_DIR = os.path.join(
    DATA_DIR,
    "predictions"
)

OUT_BASE_DIR = os.path.join(
    BASE_DIR,
    "reconstructions"
)

AUDIO_DIR = os.getenv("AUDIO_DIR")

# Se AUDIO_DIR estiver definido em variável de ambiente,
# utiliza ele. Caso contrário, coloque o caminho diretamente.
if AUDIO_DIR is None:
    AUDIO_DIR = "/home/a_matheusteixeira/dados_RosaGLM_ConservaSom_20241104/wavs_20241104"


# =========================================================
# PARÂMETROS DO ESPECTROGRAMA
# =========================================================

SR = 32768
N_FFT = 2048
HOP = 1638
N_MELS = 128

IOU_THRESHOLD = 0.3

N_FOLDS = 5


# =========================================================
# ENCONTRAR ÁUDIOS DO TESTE
# =========================================================

def get_audio_ids_from_labels(label_dir):

    audio_ids = set()

    for file in os.listdir(label_dir):

        if not file.lower().endswith(".txt"):
            continue

        audio_id = os.path.splitext(file)[0]

        audio_ids.add(audio_id)

    return sorted(audio_ids)


audio_ids_test = get_audio_ids_from_labels(GT_DIR)


print("=" * 70)
print("ÁUDIOS DO TESTE")
print("=" * 70)

print(
    f"Total de áudios encontrados: "
    f"{len(audio_ids_test)}"
)

for audio_id in audio_ids_test:
    print(audio_id)


# =========================================================
# MAPEAR WAVs
# =========================================================

audio_paths = {}

for file in os.listdir(AUDIO_DIR):

    if not file.lower().endswith(".wav"):
        continue

    audio_id = os.path.splitext(file)[0]

    audio_paths[audio_id] = os.path.join(
        AUDIO_DIR,
        file
    )


# =========================================================
# FUNÇÕES
# =========================================================

def yolo_to_timefreq(
    box,
    duration,
    sr,
    n_mels
):

    xc, yc, w, h = box

    # -----------------------------------------------------
    # TEMPO
    # -----------------------------------------------------

    ts = (
        xc - w / 2
    ) * duration

    te = (
        xc + w / 2
    ) * duration

    ts = max(0, ts)
    te = min(duration, te)

    # -----------------------------------------------------
    # POSIÇÃO VERTICAL
    # -----------------------------------------------------

    y_top = yc - h / 2
    y_bottom = yc + h / 2

    # Inverte verticalmente:
    #
    # YOLO:
    # yc = 0 -> topo
    # yc = 1 -> fundo
    #
    # Espectrograma:
    # frequência alta -> topo
    # frequência baixa -> fundo

    mel_high_norm = 1 - y_top
    mel_low_norm = 1 - y_bottom

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

    # -----------------------------------------------------
    # MEL -> HZ
    # -----------------------------------------------------

    mel_min = librosa.hz_to_mel(0)

    mel_max = librosa.hz_to_mel(
        sr / 2
    )

    mel_low = (
        mel_min
        + mel_low_norm *
        (mel_max - mel_min)
    )

    mel_high = (
        mel_min
        + mel_high_norm *
        (mel_max - mel_min)
    )

    f_low = librosa.mel_to_hz(
        mel_low
    )

    f_high = librosa.mel_to_hz(
        mel_high
    )

    return ts, te, f_low, f_high


def intersection(a, b):

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])

    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    return (
        max(0, xB - xA)
        *
        max(0, yB - yA)
    )


def iou(a, b):

    inter = intersection(a, b)

    areaA = (
        (a[2] - a[0])
        *
        (a[3] - a[1])
    )

    areaB = (
        (b[2] - b[0])
        *
        (b[3] - b[1])
    )

    union = (
        areaA
        + areaB
        - inter
    )

    return (
        inter / union
        if union > 0
        else 0
    )


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


# =========================================================
# PROCESSAR UM ÁUDIO
# =========================================================

def process_audio(
    audio_id,
    audio_path,
    pred_dir,
    out_dir
):

    tp = 0
    fp = 0
    fn = 0

    # =====================================================
    # CARREGAR ÁUDIO
    # =====================================================

    y, sr = librosa.load(
        audio_path,
        sr=SR
    )

    y = spectral_subtraction(y, sr)
    duration = len(y) / sr

    # =====================================================
    # MEL SPECTROGRAM
    # =====================================================

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

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(
        figsize=(20, 6)
    )

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

    # =====================================================
    # GT
    # =====================================================

    gt_path = os.path.join(
        GT_DIR,
        f"{audio_id}.txt"
    )

    gt_boxes = []

    if os.path.exists(gt_path):

        with open(gt_path) as f:

            for line in f:

                values = list(
                    map(
                        float,
                        line.split()
                    )
                )

                if len(values) < 5:
                    continue

                cls, xc, yc, w, h = values[:5]

                ts, te, fs, fe = (
                    yolo_to_timefreq(
                        (xc, yc, w, h),
                        duration,
                        sr,
                        N_MELS
                    )
                )

                gt_boxes.append(
                    [ts, fs, te, fe]
                )

    else:

        print(
            f"⚠️ GT não encontrado: "
            f"{gt_path}"
        )

    # =====================================================
    # PRED
    # =====================================================

    pred_path = os.path.join(
        pred_dir,
        f"{audio_id}.txt"
    )

    pred_boxes = []

    if os.path.exists(pred_path):

        with open(pred_path) as f:

            for line in f:

                values = list(
                    map(
                        float,
                        line.split()
                    )
                )

                if len(values) < 5:
                    continue

                # YOLO prediction:
                #
                # cls xc yc w h conf

                cls, xc, yc, w, h = values[:5]

                ts, te, fs, fe = (
                    yolo_to_timefreq(
                        (xc, yc, w, h),
                        duration,
                        sr,
                        N_MELS
                    )
                )

                pred_boxes.append(
                    [ts, fs, te, fe]
                )

    # =====================================================
    # MATCHING
    # =====================================================

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

            val = iou(
                p,
                g
            )

            if val > best_iou:

                best_iou = val
                best_gi = gi

        # =================================================
        # TP
        # =================================================

        if best_iou >= IOU_THRESHOLD:

            matched_gt.add(
                best_gi
            )

            matched_pred.add(
                pi
            )

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

        # =================================================
        # FP
        # =================================================

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

    # =====================================================
    # FN
    # =====================================================

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

    # =====================================================
    # TÍTULO
    # =====================================================

    plt.title(
        f"{audio_id} | "
        f"TP={tp} | "
        f"FP={fp} | "
        f"FN={fn}"
    )

    # =====================================================
    # SALVAR
    # =====================================================

    output_path = os.path.join(
        out_dir,
        f"{audio_id}_error_map.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # MÉTRICAS
    # =====================================================

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
        2 * precision * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    print(
        f"✅ {audio_id} | "
        f"TP={tp} "
        f"FP={fp} "
        f"FN={fn} | "
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


# =========================================================
# PROCESSAR TODOS OS FOLDS
# =========================================================

all_fold_results = []

for fold in range(1, N_FOLDS + 1):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    # =====================================================
    # DIRETÓRIO DAS PREDIÇÕES
    # =====================================================

    pred_dir = os.path.join(
        PRED_BASE_DIR,
        f"fold_{fold}",
        "labels"
    )

    if not os.path.exists(pred_dir):

        print(
            f"❌ Diretório de predições "
            f"não encontrado:"
        )

        print(pred_dir)

        continue

    # =====================================================
    # DIRETÓRIO DE SAÍDA
    # =====================================================

    out_dir = os.path.join(
        OUT_BASE_DIR,
        f"fold_{fold}"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # =====================================================
    # RESULTADOS DO FOLD
    # =====================================================

    fold_results = []

    # =====================================================
    # PROCESSAR ÁUDIOS
    # =====================================================

    for audio_id in audio_ids_test:

        if audio_id not in audio_paths:

            print(
                f"⚠️ WAV não encontrado: "
                f"{audio_id}"
            )

            continue

        audio_path = audio_paths[
            audio_id
        ]

        try:

            result = process_audio(
                audio_id,
                audio_path,
                pred_dir,
                out_dir
            )

            fold_results.append(
                result
            )

        except Exception as e:

            print(
                f"❌ Erro em "
                f"{audio_id}: {e}"
            )

    # =====================================================
    # DATAFRAME DO FOLD
    # =====================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    if fold_df.empty:

        print(
            f"⚠️ Nenhum áudio processado "
            f"no Fold {fold}"
        )

        continue

    # Adiciona identificação do fold

    fold_df.insert(
        0,
        "fold",
        fold
    )

    all_fold_results.append(
        fold_df
    )

    # =====================================================
    # CSV DO FOLD
    # =====================================================

    fold_csv = os.path.join(
        out_dir,
        "audio_metrics.csv"
    )

    fold_df.to_csv(
        fold_csv,
        index=False
    )

    # =====================================================
    # ESTATÍSTICAS DO FOLD
    # =====================================================

    print("\n")
    print(
        f"===== ESTATÍSTICAS FOLD {fold} ====="
    )

    print(
        f"Precision média: "
        f"{fold_df['precision'].mean():.4f}"
    )

    print(
        f"Recall médio: "
        f"{fold_df['recall'].mean():.4f}"
    )

    print(
        f"F1 médio: "
        f"{fold_df['f1'].mean():.4f}"
    )

    print(
        f"F1 desvio padrão: "
        f"{fold_df['f1'].std():.4f}"
    )

    # =====================================================
    # TOP 5
    # =====================================================

    top5 = (
        fold_df
        .sort_values(
            "f1",
            ascending=False
        )
        .head(5)
    )

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
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # BOTTOM 5
    # =====================================================

    bottom5 = (
        fold_df
        .sort_values(
            "f1",
            ascending=True
        )
        .head(5)
    )

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
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # 5 MAIS PRÓXIMOS DA MEDIANA
    # =====================================================

    median_f1 = (
        fold_df["f1"].median()
    )

    fold_df[
        "distance_to_median"
    ] = (
        fold_df["f1"]
        - median_f1
    ).abs()

    middle5 = (
        fold_df
        .sort_values(
            "distance_to_median"
        )
        .head(5)
    )

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
        ].to_string(
            index=False
        )
    )


# =========================================================
# JUNTAR RESULTADOS DE TODOS OS FOLDS
# =========================================================

if len(all_fold_results) == 0:

    print(
        "\n❌ Nenhum fold foi processado."
    )

    exit()


all_results_df = pd.concat(
    all_fold_results,
    ignore_index=True
)


# =========================================================
# CSV GERAL
# =========================================================

all_csv = os.path.join(
    OUT_BASE_DIR,
    "all_folds_audio_metrics.csv"
)

all_results_df.to_csv(
    all_csv,
    index=False
)


# =========================================================
# ESTATÍSTICAS GERAIS POR FOLD
# =========================================================

fold_summary = (
    all_results_df
    .groupby("fold")
    .agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),

        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),

        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),

        total_tp=("tp", "sum"),
        total_fp=("fp", "sum"),
        total_fn=("fn", "sum"),

        n_audios=("audio_id", "count")
    )
    .reset_index()
)


# =========================================================
# MOSTRAR RESUMO
# =========================================================

print("\n")
print("=" * 70)
print("RESUMO DOS 5 FOLDS")
print("=" * 70)

print(
    fold_summary.to_string(
        index=False
    )
)


# =========================================================
# MÉDIA E DESVIO PADRÃO ENTRE OS FOLDS
# =========================================================

print("\n")
print("=" * 70)
print("MÉDIA E DESVIO PADRÃO ENTRE OS FOLDS")
print("=" * 70)

print(
    f"Precision média: "
    f"{fold_summary['precision_mean'].mean():.4f}"
)

print(
    f"Precision std:   "
    f"{fold_summary['precision_mean'].std():.4f}"
)

print(
    f"Recall médio:    "
    f"{fold_summary['recall_mean'].mean():.4f}"
)

print(
    f"Recall std:      "
    f"{fold_summary['recall_mean'].std():.4f}"
)

print(
    f"F1 médio:        "
    f"{fold_summary['f1_mean'].mean():.4f}"
)

print(
    f"F1 std:          "
    f"{fold_summary['f1_mean'].std():.4f}"
)


# =========================================================
# MELHOR FOLD POR F1
# =========================================================

best_fold_idx = (
    fold_summary["f1_mean"]
    .idxmax()
)

best_fold = int(
    fold_summary.loc[
        best_fold_idx,
        "fold"
    ]
)

best_f1 = (
    fold_summary.loc[
        best_fold_idx,
        "f1_mean"
    ]
)


print("\n")
print("=" * 70)
print("MELHOR FOLD")
print("=" * 70)

print(
    f"Fold {best_fold}"
)

print(
    f"F1 médio: {best_f1:.4f}"
)


# =========================================================
# SALVAR RESUMO
# =========================================================

summary_csv = os.path.join(
    OUT_BASE_DIR,
    "fold_summary.csv"
)

fold_summary.to_csv(
    summary_csv,
    index=False
)


print("\n")
print("=" * 70)
print("ARQUIVOS SALVOS")
print("=" * 70)

print(
    f"Resultados por áudio:"
)

print(all_csv)

print(
    f"\nResumo dos folds:"
)

print(summary_csv)

print(
    f"\nReconstruções:"
)

print(OUT_BASE_DIR)

print("=" * 70)
