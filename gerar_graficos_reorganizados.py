
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = "/home/a_matheusteixeira/novos/"

# Ordem pensada para facilitar a comparação visual
VERSOES = [
    "v11", "v12",
    "v17", "v13", "v22", "v21",
    "v19", "v15", "v26", "v25",
    "v18", "v14", "v24", "v23",
    "v20", "v16", "v28", "v27"
]

# Nome curto para aparecer no gráfico
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
# LEITURA
# ============================================================

resultados = []

for versao in VERSOES:
    csv_path = os.path.join(
        BASE_DIR,
        versao,
        "reconstructions",
        "fold_summary.csv"
    )

    if not os.path.exists(csv_path):
        print(f"⚠️ Não encontrado: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Média dos 5 folds
    precision_mean = df["precision_mean"].mean()
    recall_mean = df["recall_mean"].mean()
    f1_mean = df["f1_mean"].mean()

    # Melhor fold = maior F1
    melhor_fold = df.loc[df["f1_mean"].idxmax()]

    resultados.append({
        "versao": versao,
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "f1_mean": f1_mean,
        "best_fold": int(melhor_fold["fold"]),
        "precision_best": melhor_fold["precision_mean"],
        "recall_best": melhor_fold["recall_mean"],
        "f1_best": melhor_fold["f1_mean"]
    })

resultados_df = pd.DataFrame(resultados)

# Garante a ordem definida acima
resultados_df["ordem"] = resultados_df["versao"].map(
    {v: i for i, v in enumerate(VERSOES)}
)

resultados_df = resultados_df.sort_values("ordem").reset_index(drop=True)

# Salva tabela consolidada
resultados_df.to_csv(
    os.path.join(BASE_DIR, "comparacao_metricas_reorganizada.csv"),
    index=False
)

# ============================================================
# FUNÇÃO PARA ADICIONAR SEPARADORES DE GRUPOS
# ============================================================

def adicionar_grupos():
    # Posições após cada grupo
    separadores = [1, 5, 9, 13, 17]

    for pos in separadores:
        plt.axvline(
            pos - 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )

# ============================================================
# GRÁFICO 1 — MÉDIA DOS FOLDS
# ============================================================

x = range(len(resultados_df))

plt.figure(figsize=(20, 8))

plt.plot(
    x,
    resultados_df["precision_mean"],
    marker="o",
    linewidth=2,
    label="Precisão média"
)

plt.plot(
    x,
    resultados_df["recall_mean"],
    marker="o",
    linewidth=2,
    label="Recall médio"
)

plt.plot(
    x,
    resultados_df["f1_mean"],
    marker="o",
    linewidth=2,
    label="F1-score médio"
)

adicionar_grupos()

plt.xticks(
    x,
    [NAMES[v] for v in resultados_df["versao"]],
    rotation=0,
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
    os.path.join(BASE_DIR, "metricas_medias_reorganizadas.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# GRÁFICO 2 — MELHOR FOLD
# ============================================================

plt.figure(figsize=(20, 8))

plt.plot(
    x,
    resultados_df["precision_best"],
    marker="o",
    linewidth=2,
    label="Precisão — melhor fold"
)

plt.plot(
    x,
    resultados_df["recall_best"],
    marker="o",
    linewidth=2,
    label="Recall — melhor fold"
)

plt.plot(
    x,
    resultados_df["f1_best"],
    marker="o",
    linewidth=2,
    label="F1-score — melhor fold"
)

adicionar_grupos()

plt.xticks(
    x,
    [NAMES[v] for v in resultados_df["versao"]],
    rotation=0,
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
    os.path.join(BASE_DIR, "melhor_fold_reorganizado.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# TABELA FINAL
# ============================================================

print("\n" + "=" * 100)
print("COMPARAÇÃO REORGANIZADA")
print("=" * 100)

print(
    resultados_df[
        [
            "versao",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "best_fold",
            "precision_best",
            "recall_best",
            "f1_best"
        ]
    ].to_string(index=False)
)

