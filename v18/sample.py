import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))

coluna_especie = "roi_label"
coluna_anotador = "roi_user"
coluna_valor = "roi_duration"

# Quantidade original de espécies
especies_originais = df[coluna_especie].nunique()

# Calcular P10 e P90 da duração por anotador
percentis = (
    df.groupby(coluna_anotador)[coluna_valor]
      .quantile([0.10, 0.90])
      .unstack()
      .rename(columns={0.10: "P10", 0.90: "P90"})
      .reset_index()
)

# Juntar percentis ao dataframe original
df = df.merge(percentis, on=coluna_anotador, how="left")

# Manter anotações entre P10 e P90
df_filtrado = df[
    (df[coluna_valor] >= df["P10"]) &
    (df[coluna_valor] <= df["P90"])
].copy()

# Remover colunas auxiliares
df_filtrado.drop(columns=["P10", "P90"], inplace=True)

# Quantidade de espécies após a filtragem
especies_finais = df_filtrado[coluna_especie].nunique()

# Identificar espécies que ficaram sem anotações
especies_removidas = (
    set(df[coluna_especie].dropna().unique())
    - set(df_filtrado[coluna_especie].dropna().unique())
)

# ---------------- DEBUG POR ANOTADOR ----------------
print("\n===== Percentis por anotador =====\n")

for _, row in percentis.iterrows():
    anotador = row[coluna_anotador]

    original = (df[coluna_anotador] == anotador).sum()
    final = (df_filtrado[coluna_anotador] == anotador).sum()

    print(
        f"Anotador: {str(anotador):20} "
        f"P10={row['P10']:.3f} "
        f"P90={row['P90']:.3f} "
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
saida = "anotacoes_p10_p90.csv"
df_filtrado.to_csv(saida, index=False)

print(f"\nArquivo salvo em: {os.path.abspath(saida)}")
