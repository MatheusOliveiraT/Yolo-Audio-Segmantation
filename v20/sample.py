import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))  # ou pd.read_csv("anotacoes.csv")

coluna_especie = "roi_label"
coluna_anotador = "roi_user"
coluna_valor = "roi_duration"

# Quantidade original de espécies
especies_originais = df[coluna_especie].nunique()

# Calcula P10 e P90 da duração para cada anotador
percentis = (
    df.groupby(coluna_anotador)[coluna_valor]
      .quantile([0.25, 0.75])
      .unstack()
      .rename(columns={0.25: "P25", 0.75: "P75"})
      .reset_index()
)

# Junta os percentis ao dataframe original
df = df.merge(percentis, on=coluna_anotador, how="left")

# Mantém apenas as anotações entre P10 e P90
df_filtrado = df[
    (df[coluna_valor] >= df["P25"]) &
    (df[coluna_valor] <= df["P75"])
].copy()

# Remove colunas auxiliares
df_filtrado.drop(columns=["P25", "P75"], inplace=True)

# Quantidade de espécies após a filtragem
especies_finais = df_filtrado[coluna_especie].nunique()

# Identificar espécies que ficaram sem anotações
especies_removidas = (
    set(df[coluna_especie].dropna().unique())
    - set(df_filtrado[coluna_especie].dropna().unique())
)


# ---------------- DEBUG ----------------
print("\n===== Percentis por anotador =====\n")

for _, row in percentis.iterrows():
    anotador = row["roi_user"]

    original = len(df[df["roi_user"] == anotador])
    final = len(df_filtrado[df_filtrado["roi_user"] == anotador])

    print(
        f"Anotador: {anotador:20} "
        f"P25={row['P25']:.3f} "
        f"P75={row['P75']:.3f} "
        f"Original={original:5d} "
        f"Filtrado={final:5d}"
    )

# ---------------- RESUMO GERAL ----------------
print("\n===== RESUMO DA FILTRAGEM =====")
print(f"Total original de anotações: {len(df)}")
print(f"Total de anotações filtradas: {len(df_filtrado)}")
print(f"Quantidade original de espécies: {especies_originais}")
print(f"Quantidade de espécies após filtragem: {especies_finais}")
print(f"Quantidade de espécies removidas: {len(especies_removidas)}")

if especies_removidas:
    print("\nEspécies sem anotações após a filtragem:")
    for especie in sorted(especies_removidas):
        print(f"- {especie}")

# Salvar resultado
saida = "anotacoes_p25_p75.csv"
df_filtrado.to_csv(saida, index=False)

print(f"\nArquivo salvo em: {os.path.abspath(saida)}")
