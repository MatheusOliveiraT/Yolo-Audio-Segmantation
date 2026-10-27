import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))

coluna_valor = "roi_duration"
coluna_especie = "roi_label"

# Quantidade original de espécies
especies_originais = df[coluna_especie].nunique()

# Calcular P10 e P90 para cada espécie
percentis = (
    df.groupby(coluna_especie)[coluna_valor]
      .quantile([0.10, 0.90])
      .unstack()
      .rename(columns={0.10: "P10", 0.90: "P90"})
)

# Juntar os percentis ao dataframe
df = df.merge(percentis, on=coluna_especie)

# Manter registros entre P10 e P90
df_filtrado = df[
    (df[coluna_valor] >= df["P10"]) &
    (df[coluna_valor] <= df["P90"])
].copy()

# Remover colunas auxiliares
df_filtrado.drop(columns=["P10", "P90"], inplace=True)

# Quantidade de espécies após a filtragem
especies_finais = df_filtrado[coluna_especie].nunique()

# Espécies removidas
especies_removidas = set(df[coluna_especie].unique()) - set(
    df_filtrado[coluna_especie].unique()
)

# Debug por espécie
print("\n===== Percentis por espécie =====\n")

for especie, row in percentis.iterrows():
    n_original = (df[coluna_especie] == especie).sum()
    n_final = (df_filtrado[coluna_especie] == especie).sum()

    print(
        f"{especie:30} "
        f"P10={row['P10']:.2f} "
        f"P90={row['P90']:.2f} "
        f"Original={n_original} "
        f"Filtrado={n_final}"
    )

# Resumo geral
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
