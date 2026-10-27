import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))  # ou pd.read_csv("anotacoes.csv")

# Calcula P10 e P90 da duração para cada anotador
percentis = (
    df.groupby("roi_user")["roi_duration"]
      .quantile([0.10, 0.90])
      .unstack()
      .rename(columns={0.10: "P10", 0.90: "P90"})
      .reset_index()
)

# Junta os percentis ao dataframe original
df = df.merge(percentis, on="roi_user", how="left")

# Mantém apenas as anotações entre P10 e P90
df_filtrado = df[
    (df["roi_duration"] >= df["P10"]) &
    (df["roi_duration"] <= df["P90"])
].copy()

# Remove colunas auxiliares
df_filtrado.drop(columns=["P10", "P90"], inplace=True)

# ---------------- DEBUG ----------------
print("\n===== Percentis por anotador =====\n")

for _, row in percentis.iterrows():
    anotador = row["roi_user"]

    original = len(df[df["roi_user"] == anotador])
    final = len(df_filtrado[df_filtrado["roi_user"] == anotador])

    print(
        f"Anotador: {anotador:20} "
        f"P10={row['P10']:.3f} "
        f"P90={row['P90']:.3f} "
        f"Original={original:5d} "
        f"Filtrado={final:5d}"
    )

print("\n=============================")
print(f"Total original: {len(df)}")
print(f"Total filtrado: {len(df_filtrado)}")

# Salvar
df_filtrado.to_csv("anotacoes_p10_p90.csv", index=False)