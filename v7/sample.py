import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))  # ou pd.read_csv("anotacoes.csv")

# Nome da coluna que será usada para calcular os percentis
coluna_valor = "roi_duration"  # Altere para o nome correto

# Calcula P10 e P90 para cada espécie
percentis = (
    df.groupby("roi_label")[coluna_valor]
      .quantile([0.10, 0.90])
      .unstack()
      .rename(columns={0.10: "P10", 0.90: "P90"})
)

# Junta os percentis ao dataframe original
df = df.merge(percentis, on="roi_label")

# Mantém apenas os registros entre P10 e P90
df_filtrado = df[
    (df[coluna_valor] >= df["P10"]) &
    (df[coluna_valor] <= df["P90"])
].copy()

# Remove colunas auxiliares
df_filtrado.drop(columns=["P10", "P90"], inplace=True)

# Debug
print("\n===== Percentis por espécie =====\n")
for especie, row in percentis.iterrows():
    n_original = (df["roi_label"] == especie).sum()
    n_final = (df_filtrado["roi_label"] == especie).sum()

    print(
        f"{especie:30} "
        f"P10={row['P10']:.2f} "
        f"P90={row['P90']:.2f} "
        f"Original={n_original} "
        f"Filtrado={n_final}"
    )

print(f"\nTotal original: {len(df)}")
print(f"Total filtrado: {len(df_filtrado)}")

# Salvar
df_filtrado.to_csv("anotacoes_p10_p90.csv", index=False)