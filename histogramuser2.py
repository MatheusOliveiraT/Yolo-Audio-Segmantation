import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

CSV_PATH = "/home/a_matheusteixeira/df_ROI_RosaGLM_ConservaSom_20241104_certeza.csv"

OUTPUT_DIR = "/home/a_matheusteixeira/dataset_statistics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# CARREGA DATASET
# ==========================================================

df = pd.read_csv(CSV_PATH)

# ==========================================================
# CONTAGEM DE ANOTAÇÕES POR ANOTADOR
# ==========================================================

annotator_counts = (
    df["roi_user"]
    .value_counts()
    .sort_values(ascending=False)
)

print("Quantidade de anotações por anotador:\n")
print(annotator_counts)

# Salva as contagens em CSV
annotator_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "annotations_per_annotator.csv"
    ),
    header=["num_annotations"]
)

# ==========================================================
# HISTOGRAMA
# ==========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    annotator_counts.index.astype(str),
    annotator_counts.values
)

plt.xlabel("Anotador")
plt.ylabel("Número de anotações")
plt.title("Quantidade de anotações por anotador")

plt.xticks(rotation=45, ha="right")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# Adiciona o valor acima de cada barra
for i, value in enumerate(annotator_counts.values):
    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "annotations_per_annotator.png"
    ),
    dpi=300
)

plt.close()

print(f"\nNúmero de anotadores: {len(annotator_counts)}")
print(f"Total de anotações: {annotator_counts.sum()}")

print("\nHistograma salvo em:")
print(os.path.join(
    OUTPUT_DIR,
    "annotations_per_annotator.png"
))