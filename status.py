import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
csv_path = "/home/a_matheusteixeira/df_ROI_RosaGLM_ConservaSom_20241104_certeza.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(csv_path)

print("=" * 80)
print("VISÃO GERAL DO DATASET")
print("=" * 80)

print(f"\nNúmero total de anotações: {len(df):,}")

# =========================
# CLASSES / ESPÉCIES
# =========================
print("\n" + "=" * 80)
print("CLASSES (ESPÉCIES)")
print("=" * 80)

n_classes = df["roi_label"].nunique()

print(f"\nNúmero de classes (espécies): {n_classes}")

dist_classes = (
    df["roi_label"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nTop 20 espécies mais frequentes:\n")
print(dist_classes.head(20))

print("\nTop 20 espécies menos frequentes:\n")
print(dist_classes.tail(20))

print("\nEstatísticas da distribuição de exemplos por classe:")
print(dist_classes.describe())

# =========================
# ANOTADORES
# =========================
print("\n" + "=" * 80)
print("ANOTADORES")
print("=" * 80)

n_annotators = df["roi_user"].nunique()

print(f"\nNúmero de anotadores: {n_annotators}")

annotator_dist = (
    df["roi_user"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nQuantidade de anotações por anotador:\n")
print(annotator_dist)

# Quantidade média de espécies por anotador
species_per_annotator = (
    df.groupby("roi_user")["roi_label"]
    .nunique()
)

print("\nEspécies diferentes anotadas por anotador:")
print(species_per_annotator)

# =========================
# LOCALIZAÇÕES
# =========================
print("\n" + "=" * 80)
print("LOCALIZAÇÕES DAS GRAVAÇÕES")
print("=" * 80)

if "area" in df.columns:
    print("\nDistribuição por área:\n")
    print(df["area"].value_counts())

if "point" in df.columns:
    print("\nDistribuição por ponto de coleta:\n")
    print(df["point"].value_counts())

if "environment" in df.columns:
    print("\nDistribuição por ambiente:\n")
    print(df["environment"].value_counts())

# Coordenadas
if "lat" in df.columns and "long" in df.columns:
    print("\nResumo das coordenadas:")
    print(df[["lat", "long"]].describe())

# =========================
# CARACTERÍSTICAS TÉCNICAS
# =========================
print("\n" + "=" * 80)
print("CARACTERÍSTICAS TÉCNICAS")
print("=" * 80)

# Sample rate
if "roi_sample_rate" in df.columns:
    print("\nTaxas de amostragem encontradas:")
    print(df["roi_sample_rate"].value_counts())

# Duração
if "roi_duration" in df.columns:
    durations = df["roi_duration"].dropna()

    print("\nEstatísticas de duração das anotações (segundos):")
    print(durations.describe())

# Frequências
if "roi_min_freq" in df.columns and "roi_max_freq" in df.columns:
    print("\nFaixas de frequência:")

    print("\nFrequência mínima:")
    print(df["roi_min_freq"].describe())

    print("\nFrequência máxima:")
    print(df["roi_max_freq"].describe())

# =========================
# COMPLETUDE DAS ANOTAÇÕES
# =========================
print("\n" + "=" * 80)
print("COMPLETUDE DAS ANOTAÇÕES")
print("=" * 80)

if "roi_is_complete" in df.columns:
    print("\nAnotações completas vs incompletas:")
    print(df["roi_is_complete"].value_counts(dropna=False))

# =========================
# CONFIANÇA DOS RÓTULOS
# =========================
print("\n" + "=" * 80)
print("CONFIANÇA DOS RÓTULOS")
print("=" * 80)

if "roi_label_confidence" in df.columns:
    print("\nDistribuição da confiança:")
    print(df["roi_label_confidence"].describe())

# =========================
# DISTRIBUIÇÃO TEMPORAL
# =========================
print("\n" + "=" * 80)
print("DISTRIBUIÇÃO TEMPORAL")
print("=" * 80)

if "Year" in df.columns:
    print("\nDistribuição por ano:")
    print(df["Year"].value_counts().sort_index())

if "Month" in df.columns:
    print("\nDistribuição por mês:")
    print(df["Month"].value_counts().sort_index())

if "Period" in df.columns:
    print("\nDistribuição por período do dia:")
    print(df["Period"].value_counts())

if "Hour" in df.columns:
    print("\nDistribuição por hora:")
    print(df["Hour"].value_counts().sort_index())

# =========================
# TIPOS DE ROI
# =========================
print("\n" + "=" * 80)
print("TIPOS DE ROI")
print("=" * 80)

if "roi_type" in df.columns:
    print("\nTipos de ROI:")
    print(df["roi_type"].value_counts())

# =========================
# OVERLAP E WL
# =========================
print("\n" + "=" * 80)
print("PARÂMETROS DE PROCESSAMENTO")
print("=" * 80)

if "roi_wl" in df.columns:
    print("\nWindow length (roi_wl):")
    print(df["roi_wl"].describe())

if "roi_ovlp" in df.columns:
    print("\nOverlap (roi_ovlp):")
    print(df["roi_ovlp"].describe())

# =========================
# DATASET SUMMARY
# =========================
print("\n" + "=" * 80)
print("RESUMO FINAL")
print("=" * 80)

print(f"""
Resumo do dataset:

- Total de anotações: {len(df):,}
- Número de espécies: {n_classes}
- Número de anotadores: {n_annotators}
- Média de exemplos por espécie: {dist_classes.mean():.2f}
- Mediana de exemplos por espécie: {dist_classes.median():.2f}
- Classe mais frequente: {dist_classes.idxmax()} ({dist_classes.max()} exemplos)
- Classe menos frequente: {dist_classes.idxmin()} ({dist_classes.min()} exemplos)
""")
