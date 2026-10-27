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
# CONTAGEM DE ANOTAÇÕES POR ESPÉCIE
# ==========================================================

species_counts = (
    df["roi_label"]
    .value_counts()
    .sort_values(ascending=False)
)

print(species_counts)

# Salva também em CSV
species_counts.to_csv(
    os.path.join(OUTPUT_DIR, "species_annotation_counts.csv"),
    header=["num_annotations"]
)

# ==========================================================
# HISTOGRAMA
# ==========================================================

plt.figure(figsize=(18, 7))

plt.bar(
    species_counts.index,
    species_counts.values
)

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.ylabel("Número de anotações")
plt.xlabel("Espécie")
plt.title("Distribuição de anotações por espécie")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "species_annotation_histogram.png"
    ),
    dpi=300
)

plt.close()

print(f"\nNúmero de espécies: {len(species_counts)}")
print(f"Total de anotações: {species_counts.sum()}")

print("\nTop 10 espécies mais anotadas:")
print(species_counts.head(10))