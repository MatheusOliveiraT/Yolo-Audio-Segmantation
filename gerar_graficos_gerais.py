import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = "/home/a_matheusteixeira/novos/"

# ============================================================
# ORDEM DAS VERSÕES
# Organizada para facilitar a comparação entre abordagens
# ============================================================

VERSIONS = [
    "v11", "v12",

    # Espécie P10-P90
    "v17", "v13", "v22", "v21",

    # Espécie P25-P75
    "v19", "v15", "v26", "v25",

    # Anotador P10-P90
    "v18", "v14", "v24", "v23",

    # Anotador P25-P75
    "v20", "v16", "v28", "v27"
]

# ============================================================
# NOMES PARA VISUALIZAÇÃO
# ============================================================

NAMES = {
    "v11": "Base sem redu.\n ruído",
    "v12": "Base com redu.\n ruído",

    "v17": "Esp. P10-P90\nsem redu.\n ruído",
    "v13": "Esp. P10-P90\ncom redu.\n ruído",
    "v22": "Esp. P10-P90\nsem redu.\n ruído + ≥5",
    "v21": "Esp. P10-P90\ncom redu.\n ruído + ≥5",

    "v19": "Esp. P25-P75\nsem redu.\n ruído",
    "v15": "Esp. P25-P75\ncom redu.\n ruído",
    "v26": "Esp. P25-P75\nsem redu.\n ruído + ≥5",
    "v25": "Esp. P25-P75\ncom redu.\n ruído + ≥5",

    "v18": "Anot. P10-P90\nsem redu.\n ruído",
    "v14": "Anot. P10-P90\ncom redu.\n ruído",
    "v24": "Anot. P10-P90\nsem redu.\n ruído + ≥5",
    "v23": "Anot. P10-P90\ncom redu.\n ruído + ≥5",

    "v20": "Anot. P25-P75\nsem redu.\n ruído",
    "v16": "Anot. P25-P75\ncom redu.\n ruído",
    "v28": "Anot. P25-P75\nsem redu.\n ruído + ≥5",
    "v27": "Anot. P25-P75\ncom redu.\n ruído + ≥5"
}

# ============================================================
# LEITURA DOS RESULTADOS
# ============================================================

results = []

for version in VERSIONS:

    # --------------------------------------------------------
    # Localização do CSV de Precision / Recall / F1
    # --------------------------------------------------------

    fold_csv = os.path.join(
        BASE_DIR,
        version,
        "reconstructions",
        "fold_summary.csv"
    )

    # --------------------------------------------------------
    # Localização do CSV de IoU / Dice
    # --------------------------------------------------------

    metric_csv = os.path.join(
        BASE_DIR,
        version,
        "test_folds_results.csv"
    )

    # --------------------------------------------------------
    # Verificação dos arquivos
    # --------------------------------------------------------

    if not os.path.exists(fold_csv):
        print(f"⚠️ Fold summary não encontrado: {fold_csv}")
        continue

    if not os.path.exists(metric_csv):
        print(f"⚠️ IoU/Dice não encontrado: {metric_csv}")
        continue

    # =========================
    # PRECISION / RECALL / F1
    # =========================

    fold_df = pd.read_csv(fold_csv)

    precision_mean = fold_df["precision_mean"].mean()
    recall_mean = fold_df["recall_mean"].mean()
    f1_mean = fold_df["f1_mean"].mean()

    precision_std = fold_df["precision_mean"].std()
    recall_std = fold_df["recall_mean"].std()
    f1_std = fold_df["f1_mean"].std()

    best_idx = fold_df["f1_mean"].idxmax()
    best_fold = fold_df.loc[best_idx, "fold"]

    precision_best = fold_df.loc[best_idx, "precision_mean"]
    recall_best = fold_df.loc[best_idx, "recall_mean"]
    f1_best = fold_df.loc[best_idx, "f1_mean"]


    # =========================
    # IOU / DICE
    # =========================

    metric_df = pd.read_csv(metric_csv)

    iou_col = next(
        (c for c in metric_df.columns
         if "iou" in c.lower() and "mean" in c.lower()),
        None
    )

    dice_col = next(
        (c for c in metric_df.columns
         if "dice" in c.lower() and "mean" in c.lower()),
        None
    )

    # Caso as colunas sejam simplesmente "iou" / "dice"
    if iou_col is None:
        iou_col = next(
            (c for c in metric_df.columns if c.lower() == "iou"),
            None
        )

    if dice_col is None:
        dice_col = next(
            (c for c in metric_df.columns if c.lower() == "dice"),
            None
        )

    if iou_col is None or dice_col is None:
        print(f"⚠️ Não encontrei IoU/Dice em {metric_csv}")
        continue

    iou_mean = metric_df[iou_col].mean()
    dice_mean = metric_df[dice_col].mean()

    iou_std = metric_df[iou_col].std()
    dice_std = metric_df[dice_col].std()

    iou_max = metric_df[iou_col].max()
    dice_max = metric_df[dice_col].max()

    # Caso as colunas sejam simplesmente "iou" / "dice"
    if iou_col is None:
        iou_col = next(
            (
                col for col in metric_df.columns
                if col.lower() == "iou"
            ),
            None
        )

    if dice_col is None:
        dice_col = next(
            (
                col for col in metric_df.columns
                if col.lower() == "dice"
            ),
            None
        )

    if iou_col is None or dice_col is None:
        print(
            f"⚠️ Não encontrei IoU/Dice em {metric_csv}"
        )
        continue

    # Média dos folds
    iou_mean = metric_df[iou_col].mean()
    dice_mean = metric_df[dice_col].mean()

    # Maior valor entre os folds
    iou_max = metric_df[iou_col].max()
    dice_max = metric_df[dice_col].max()

    # ========================================================
    # SALVA RESULTADOS
    # ========================================================

    results.append({
        "version": version,

        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,

        "best_fold": best_fold,
        "precision_best": precision_best,
        "recall_best": recall_best,
        "f1_best": f1_best,

        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "iou_max": iou_max,

        "dice_mean": dice_mean,
        "dice_std": dice_std,
        "dice_max": dice_max
    })

# ============================================================
# DATAFRAME FINAL
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError(
        "Nenhum resultado foi encontrado."
    )

# Garante a ordem definida em VERSIONS
results_df["order"] = results_df["version"].map(
    {version: i for i, version in enumerate(VERSIONS)}
)

results_df = (
    results_df
    .sort_values("order")
    .drop(columns="order")
    .reset_index(drop=True)
)

print("\n" + "=" * 100)
print("RESULTADOS CONSOLIDADOS")
print("=" * 100)

print(
    results_df.to_string(index=False)
)

# ============================================================
# FUNÇÃO — SEPARADORES DOS GRUPOS
# ============================================================

def add_group_separators():

    # Depois de:
    # Base
    # Espécie P10-P90
    # Espécie P25-P75
    # Anotador P10-P90
    # Anotador P25-P75

    separators = [2, 6, 10, 14]

    for position in separators:
        plt.axvline(
            position - 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )

# ============================================================
# EIXO X
# ============================================================

x = range(len(results_df))

x_labels = [
    NAMES[version]
    for version in results_df["version"]
]

# ============================================================
# 1 — PRECISION / RECALL / F1 MÉDIOS
# ============================================================

plt.figure(figsize=(20, 8))

plt.plot(
    x,
    results_df["precision_mean"],
    marker="o",
    linewidth=2,
    label="Precisão média"
)

plt.plot(
    x,
    results_df["recall_mean"],
    marker="o",
    linewidth=2,
    label="Recall médio"
)

plt.plot(
    x,
    results_df["f1_mean"],
    marker="o",
    linewidth=2,
    label="F1-score médio"
)

add_group_separators()

plt.xticks(
    x,
    x_labels,
    fontsize=9
)

plt.xlabel("Abordagem")
plt.ylabel("Métrica")
# (sem título: fica na \caption do LaTeX)

plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "precision_recall_f1_medios.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 2 — PRECISION / RECALL / F1 DO MELHOR FOLD
# ============================================================

plt.figure(figsize=(20, 8))

plt.plot(
    x,
    results_df["precision_best"],
    marker="o",
    linewidth=2,
    label="Precisão — melhor fold"
)

plt.plot(
    x,
    results_df["recall_best"],
    marker="o",
    linewidth=2,
    label="Recall — melhor fold"
)

plt.plot(
    x,
    results_df["f1_best"],
    marker="o",
    linewidth=2,
    label="F1-score — melhor fold"
)

add_group_separators()

plt.xticks(
    x,
    x_labels,
    fontsize=9
)

plt.xlabel("Abordagem")
plt.ylabel("Métrica")
# (sem título: fica na \caption do LaTeX)

plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "precision_recall_f1_melhor_fold.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 3 — IoU / DICE MÉDIOS
# ============================================================

plt.figure(figsize=(20, 8))

plt.plot(
    x,
    results_df["iou_mean"],
    marker="o",
    linewidth=2,
    label="IoU médio"
)

plt.plot(
    x,
    results_df["dice_mean"],
    marker="o",
    linewidth=2,
    label="DSC médio"
)

add_group_separators()

plt.xticks(
    x,
    x_labels,
    fontsize=9
)

plt.xlabel("Abordagem")
plt.ylabel("Métrica")
# (sem título: fica na \caption do LaTeX)

plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "iou_dice_medios.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 4 — MAIOR IoU / DICE
# ============================================================

plt.figure(figsize=(20, 8))

plt.plot(
    x,
    results_df["iou_max"],
    marker="o",
    linewidth=2,
    label="Maior IoU"
)

plt.plot(
    x,
    results_df["dice_max"],
    marker="o",
    linewidth=2,
    label="Maior DSC"
)

add_group_separators()

plt.xticks(
    x,
    x_labels,
    fontsize=9
)

plt.xlabel("Abordagem")
plt.ylabel("Métrica")
# (sem título: fica na \caption do LaTeX)

plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "iou_dice_maiores.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 5 — COMPARAÇÃO DAS CINCO MÉTRICAS
# ============================================================

plt.figure(figsize=(18, 8))

plt.errorbar(
    x, results_df["precision_mean"],
    yerr=results_df["precision_std"],
    marker="o", capsize=4, label="Precisão"
)

plt.errorbar(
    x, results_df["recall_mean"],
    yerr=results_df["recall_std"],
    marker="o", capsize=4, label="Recall"
)

plt.errorbar(
    x, results_df["f1_mean"],
    yerr=results_df["f1_std"],
    marker="o", capsize=4, label="F1-score"
)

plt.errorbar(
    x, results_df["iou_mean"],
    yerr=results_df["iou_std"],
    marker="o", capsize=4, label="IoU"
)

plt.errorbar(
    x, results_df["dice_mean"],
    yerr=results_df["dice_std"],
    marker="o", capsize=4, label="DSC"
)

add_group_separators()

plt.xticks(x, x_labels, rotation=45, ha="right")
plt.ylabel("Valor")
plt.xlabel("Abordagem")
# (sem título: fica na \caption do LaTeX)
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "cinco_metricas_medias_desvio.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Separadores dos grupos
# ------------------------------------------------------------

add_group_separators()

# ------------------------------------------------------------
# Eixo X
# ------------------------------------------------------------

plt.xticks(
    x,
    x_labels,
    fontsize=9
)

plt.xlabel("Abordagem")
plt.ylabel("Métrica")
# (sem título: fica na \caption do LaTeX)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.legend(
    loc="best"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "cinco_metricas_medias.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SALVAR TABELA CONSOLIDADA
# ============================================================

output_csv = os.path.join(
    BASE_DIR,
    "comparacao_completa_versoes.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 100)
print("ARQUIVOS GERADOS")
print("=" * 100)

print(
    f"- {os.path.join(BASE_DIR, 'precision_recall_f1_medios.png')}"
)

print(
    f"- {os.path.join(BASE_DIR, 'precision_recall_f1_melhor_fold.png')}"
)

print(
    f"- {os.path.join(BASE_DIR, 'iou_dice_medios.png')}"
)

print(
    f"- {os.path.join(BASE_DIR, 'iou_dice_maiores.png')}"
)

print(
    f"- {output_csv}"
)
