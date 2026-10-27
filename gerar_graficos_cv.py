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
    "v11": "Base sem redu. ruído",
    "v12": "Base com redu. ruído",

    "v17": "Esp. P10-P90\nsem redu. ruído",
    "v13": "Esp. P10-P90\ncom redu. ruído",
    "v22": "Esp. P10-P90\nsem redu. ruído + ≥5",
    "v21": "Esp. P10-P90\ncom redu. ruído + ≥5",

    "v19": "Esp. P25-P75\nsem redu. ruído",
    "v15": "Esp. P25-P75\ncom redu. ruído",
    "v26": "Esp. P25-P75\nsem redu. ruído + ≥5",
    "v25": "Esp. P25-P75\ncom redu. ruído + ≥5",

    "v18": "Anot. P10-P90\nsem redu. ruído",
    "v14": "Anot. P10-P90\ncom redu. ruído",
    "v24": "Anot. P10-P90\nsem redu. ruído + ≥5",
    "v23": "Anot. P10-P90\ncom redu. ruído + ≥5",

    "v20": "Anot. P25-P75\nsem redu. ruído",
    "v16": "Anot. P25-P75\ncom redu. ruído",
    "v28": "Anot. P25-P75\nsem redu. ruído + ≥5",
    "v27": "Anot. P25-P75\ncom redu. ruído + ≥5"
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

    # Média e desvio padrão das contagens de TP, FP e FN entre os folds
    contagens = {}
    for coluna in ["total_tp", "total_fp", "total_fn"]:
        if coluna in fold_df.columns:
            contagens[f"{coluna}_mean"] = fold_df[coluna].mean()
            contagens[f"{coluna}_std"] = fold_df[coluna].std()
        else:
            print(f"⚠️ Coluna '{coluna}' não encontrada em {fold_csv}")
            contagens[f"{coluna}_mean"] = float("nan")
            contagens[f"{coluna}_std"] = float("nan")

    # Total de anotações por fold: TP + FP + FN.
    # O desvio padrão é calculado sobre os totais de cada fold.
    if all(col in fold_df.columns for col in ["total_tp", "total_fp", "total_fn"]):
        total_anotacoes_por_fold = (
            fold_df["total_tp"] + fold_df["total_fp"] + fold_df["total_fn"]
        )
        total_annotations_mean = total_anotacoes_por_fold.mean()
        total_annotations_std = total_anotacoes_por_fold.std()
        total_annotations_sum = total_anotacoes_por_fold.sum()

        # Proporção percentual de TP, FP e FN no total (TP + FP + FN = 100%).
        # Calculada a partir das médias das contagens dos folds.
        total_contagens_mean = sum(
            contagens[f"{coluna}_mean"]
            for coluna in ["total_tp", "total_fp", "total_fn"]
        )
        if pd.notna(total_contagens_mean) and total_contagens_mean > 0:
            total_tp_proportion_pct = contagens["total_tp_mean"] / total_contagens_mean * 100
            total_fp_proportion_pct = contagens["total_fp_mean"] / total_contagens_mean * 100
            total_fn_proportion_pct = contagens["total_fn_mean"] / total_contagens_mean * 100
        else:
            total_tp_proportion_pct = float("nan")
            total_fp_proportion_pct = float("nan")
            total_fn_proportion_pct = float("nan")
    else:
        total_annotations_mean = float("nan")
        total_annotations_std = float("nan")
        total_annotations_sum = float("nan")
        total_tp_proportion_pct = float("nan")
        total_fp_proportion_pct = float("nan")
        total_fn_proportion_pct = float("nan")
        print(f"⚠️ Não foi possível calcular o total de anotações em {fold_csv}")

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

        # Estatísticas das contagens no fold_summary.csv
        "total_tp_mean": contagens["total_tp_mean"],
        "total_tp_std": contagens["total_tp_std"],
        "total_fp_mean": contagens["total_fp_mean"],
        "total_fp_std": contagens["total_fp_std"],
        "total_fn_mean": contagens["total_fn_mean"],
        "total_fn_std": contagens["total_fn_std"],
        "total_annotations_mean": total_annotations_mean,
        "total_annotations_std": total_annotations_std,
        "total_annotations_sum_all_folds": total_annotations_sum,
        "total_tp_proportion_pct": total_tp_proportion_pct,
        "total_fp_proportion_pct": total_fp_proportion_pct,
        "total_fn_proportion_pct": total_fn_proportion_pct,

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
# GRUPOS DE VERSÕES
# ============================================================

GROUPS = {
    "base": {
        "title": "Base",
        "versions": ["v11", "v12"],
        "filename": "metricas_base.png",
    },
    "especie_p10_p90": {
        "title": "Espécie P10-P90",
        "versions": ["v17", "v13", "v22", "v21"],
        "filename": "metricas_especie_p10_p90.png",
    },
    "especie_p25_p75": {
        "title": "Espécie P25-P75",
        "versions": ["v19", "v15", "v26", "v25"],
        "filename": "metricas_especie_p25_p75.png",
    },
    "anotador_p10_p90": {
        "title": "Anotador P10-P90",
        "versions": ["v18", "v14", "v24", "v23"],
        "filename": "metricas_anotador_p10_p90.png",
    },
    "anotador_p25_p75": {
        "title": "Anotador P25-P75",
        "versions": ["v20", "v16", "v28", "v27"],
        "filename": "metricas_anotador_p25_p75.png",
    },
}

# ============================================================
# GERAR UM GRÁFICO PARA CADA GRUPO
# Cada gráfico mostra as cinco métricas (média ± desvio padrão)
# ============================================================

metricas = [
    ("precision_mean", "precision_std", "Precisão"),
    ("recall_mean", "recall_std", "Recall"),
    ("f1_mean", "f1_std", "F1-score"),
    ("iou_mean", "iou_std", "IoU"),
    ("dice_mean", "dice_std", "DSC"),
]

arquivos_graficos = []

for group_key, group_info in GROUPS.items():
    group_df = results_df[
        results_df["version"].isin(group_info["versions"])
    ].copy()

    # Preserva a ordem definida para esse grupo
    ordem = {version: i for i, version in enumerate(group_info["versions"])}
    group_df["group_order"] = group_df["version"].map(ordem)
    group_df = (
        group_df.sort_values("group_order")
        .drop(columns="group_order")
        .reset_index(drop=True)
    )

    if group_df.empty:
        print(f"⚠️ Nenhum resultado encontrado para o grupo: {group_info['title']}")
        continue

    x = range(len(group_df))
    x_labels = [NAMES.get(version, version) for version in group_df["version"]]

    plt.figure(figsize=(max(10, len(group_df) * 3.2), 7))

    for mean_col, std_col, label in metricas:
        # Desvio padrão pode estar ausente em alguns arquivos; nesse caso,
        # desenha as médias sem barras de erro.
        yerr = group_df[std_col] if std_col in group_df.columns else None

        plt.errorbar(
            x,
            group_df[mean_col],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )

    plt.xticks(x, x_labels, fontsize=9)
    plt.xlabel("Abordagem")
    plt.ylabel("Valor da métrica")
    # Sem título: ele fica na legenda (\caption) do LaTeX
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path = os.path.join(BASE_DIR, group_info["filename"])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    arquivos_graficos.append(output_path)
    print(f"Gráfico salvo: {output_path}")

# ============================================================
# SALVAR TABELA CONSOLIDADA
# ============================================================

output_csv = os.path.join(
    BASE_DIR,
    "comparacao_completa_versoes.csv"
)
results_df.to_csv(output_csv, index=False)

# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 100)
print("ARQUIVOS GERADOS")
print("=" * 100)

for caminho in arquivos_graficos:
    print(f"- {caminho}")

print(f"- {output_csv}")
