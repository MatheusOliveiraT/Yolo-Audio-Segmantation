import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))  # ou pd.read_csv("anotacoes.csv")

# ============================================================
# 1. P25-P75 POR ESPECIE
# ============================================================

percentis = (
    df.groupby("roi_label")["roi_duration"]
      .quantile([0.25, 0.75])
      .unstack()
      .rename(columns={0.25: "P25", 0.75: "P75"})
      .reset_index()
)

# Adiciona P25 e P75 a cada anotação
df = df.merge(
    percentis,
    on="roi_label",
    how="left"
)

# Filtra duração entre P25 e P75
df_filtrado = df[
    (df["roi_duration"] >= df["P25"]) &
    (df["roi_duration"] <= df["P75"])
].copy()

# Remove colunas auxiliares
df_filtrado.drop(
    columns=["P25", "P75"],
    inplace=True
)


# ============================================================
# 2. MANTER ESPÉCIES COM PELO MENOS 5 INSTÂNCIAS
# ============================================================

contagem_especies = (
    df_filtrado["roi_label"]
    .value_counts()
)

# Espécies que possuem pelo menos 5 anotações
especies_validas = contagem_especies[
    contagem_especies >= 5
].index

# Filtra o dataframe
df_final = df_filtrado[
    df_filtrado["roi_label"].isin(especies_validas)
].copy()


# ============================================================
# DEBUG
# ============================================================

print("\n===== P25-P75 POR ESPECIE =====\n")

for _, row in percentis.iterrows():

    especie = row["roi_label"]

    original = len(
        df[df["roi_label"] == especie]
    )

    filtrado = len(
        df_filtrado[df_filtrado["roi_label"] == especie]
    )

    print(
        f"Especie: {especie:20} "
        f"P25={row['P25']:.3f} "
        f"P75={row['P75']:.3f} "
        f"Original={original:5d} "
        f"P25-P75={filtrado:5d}"
    )


print("\n===== ESPÉCIES APÓS P25-P75 =====\n")

for especie, quantidade in contagem_especies.sort_values(
    ascending=False
).items():

    status = "MANTIDA" if quantidade >= 5 else "REMOVIDA"

    print(
        f"{especie:35} "
        f"Quantidade={quantidade:5d} "
        f"{status}"
    )


# ============================================================
# RESUMO
# ============================================================

print("\n========================================")
print("RESUMO")
print("========================================")

print(f"Total original:        {len(df):6d}")
print(f"Após P25-P75:          {len(df_filtrado):6d}")
print(f"Após mínimo de 5:      {len(df_final):6d}")

print(
    f"Espécies antes:        "
    f"{df['roi_label'].nunique():6d}"
)

print(
    f"Espécies após filtro:  "
    f"{df_final['roi_label'].nunique():6d}"
)


# ============================================================
# SALVAR
# ============================================================

df_final.to_csv("anotacoes_p25_p75.csv", index=False)

print("\nArquivo salvo como: anotacoes_p25_p75.csv")