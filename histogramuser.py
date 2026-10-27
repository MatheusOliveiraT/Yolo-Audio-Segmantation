import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

CSV_PATH = "/home/a_matheusteixeira/df_ROI_RosaGLM_ConservaSom_20241104_certeza.csv"

OUTPUT_DIR = "/home/a_matheusteixeira/histogramas_anotadores"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# CARREGA DATASET
# ==========================================================

df = pd.read_csv(CSV_PATH)

# ==========================================================
# GERA UM HISTOGRAMA PARA CADA ANOTADOR
# ==========================================================

annotators = sorted(df["roi_user"].dropna().unique())

print(f"Encontrados {len(annotators)} anotadores.")

for annotator in annotators:

    df_user = df[df["roi_user"] == annotator]

    species_counts = (
        df_user["roi_label"]
        .value_counts()
        .sort_values(ascending=False)
    )

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
    plt.title(f"Distribuição de anotações por espécie\nAnotador: {annotator}")

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    filename = str(annotator).replace("/", "_").replace(" ", "_")

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{filename}.png"
        ),
        dpi=300
    )

    plt.close()

    # Também salva as contagens em CSV
    species_counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{filename}.csv"
        ),
        header=["num_annotations"]
    )

print(f"\nHistogramas salvos em:\n{OUTPUT_DIR}")