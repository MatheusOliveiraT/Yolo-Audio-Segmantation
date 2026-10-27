import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))  # ou pd.read_csv("anotacoes.csv")

# Nome da coluna que será usada para calcular os percentis
coluna_valor = "roi_duration"  # Altere para o nome correto
coluna_especie = "roi_label"

# Quantidade original de espécies
especies_originais = df[coluna_especie].nunique()

# Calcula P10 e P90 para cada espécie
percentis = (
    df.groupby("roi_label")[coluna_valor]
      .quantile([0.25, 0.75])
      .unstack()
      .rename(columns={0.25: "P25", 0.75: "P75"})
)

# Junta os percentis ao dataframe original
df = df.merge(percentis, on="roi_label")

# Mantém apenas os registros entre P10 e P90
df_filtrado = df[
    (df[coluna_valor] >= df["P25"]) &
    (df[coluna_valor] <= df["P75"])
].copy()

# Remove colunas auxiliares
df_filtrado.drop(columns=["P25", "P75"], inplace=True)

# Quantidade de espécies após a filtragem
especies_finais = df_filtrado[coluna_especie].nunique()

# Espécies removidas
especies_removidas = set(df[coluna_especie].unique()) - set(
    df_filtrado[coluna_especie].unique()
)

# Debug
print("\n===== Percentis por espécie =====\n")
for especie, row in percentis.iterrows():
    n_original = (df["roi_label"] == especie).sum()
    n_final = (df_filtrado["roi_label"] == especie).sum()

    print(
        f"{especie:30} "
        f"P25={row['P25']:.2f} "
        f"P75={row['P75']:.2f} "
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

# Salvar
saida = "anotacoes_p25_p75.csv"
df_filtrado.to_csv(saida, index=False)

print(f"\nArquivo salvo em: {os.path.abspath(saida)}")
