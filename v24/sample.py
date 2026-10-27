import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. LEITURA DOS DADOS
# ============================================================

df = pd.read_csv(os.getenv("CSV_PATH"))

coluna_anotador = "roi_user"
coluna_valor = "roi_duration"
coluna_especie = "roi_label"

# Quantidades originais
total_original = len(df)
especies_originais = df[coluna_especie].nunique()

# ============================================================
# 2. CALCULAR P10 E P90 POR ANOTADOR
# ============================================================

percentis = (
    df.groupby(coluna_anotador)[coluna_valor]
      .quantile([0.10, 0.90])
      .unstack()
      .rename(columns={0.10: "P10", 0.90: "P90"})
      .reset_index()
)

# Juntar os percentis ao dataframe original
df_com_percentis = df.merge(
    percentis,
    on=coluna_anotador,
    how="left"
)

# Manter anotações entre P10 e P90
df_filtrado = df_com_percentis[
    (df_com_percentis[coluna_valor] >= df_com_percentis["P10"]) &
    (df_com_percentis[coluna_valor] <= df_com_percentis["P90"])
].copy()

# Quantidade de espécies após P10-P90
especies_apos_percentis = df_filtrado[coluna_especie].nunique()

# ============================================================
# 3. MANTER ESPÉCIES COM PELO MENOS 5 INSTÂNCIAS
# ============================================================

contagem_especies = df_filtrado[coluna_especie].value_counts()

especies_validas = contagem_especies[
    contagem_especies >= 5
].index

df_final = df_filtrado[
    df_filtrado[coluna_especie].isin(especies_validas)
].copy()

# Quantidades finais
total_final = len(df_final)
especies_finais = df_final[coluna_especie].nunique()

# Identificar espécies removidas
especies_originais_set = set(df[coluna_especie].dropna().unique())
especies_finais_set = set(df_final[coluna_especie].dropna().unique())

especies_removidas = especies_originais_set - especies_finais_set

# ============================================================
# 4. DEBUG POR ANOTADOR
# ============================================================

print("\n===== Percentis por anotador =====\n")

for _, row in percentis.iterrows():
    anotador = row[coluna_anotador]

    original = (df[coluna_anotador] == anotador).sum()
    apos_percentis = (
        df_filtrado[coluna_anotador] == anotador
    ).sum()
    final = (df_final[coluna_anotador] == anotador).sum()

    print(
        f"Anotador: {str(anotador):20} "
        f"P10={row['P10']:.3f} "
        f"P90={row['P90']:.3f} "
        f"Original={original:5d} "
        f"Após P10-P90={apos_percentis:5d} "
        f"Final={final:5d}"
    )

# ============================================================
# 5. RESUMO GERAL
# ============================================================

print("\n===== RESUMO DA FILTRAGEM =====")

print(f"Total original de anotações: {total_original}")
print(f"Total de anotações filtradas: {total_final}")

print(f"Quantidade original de espécies: {especies_originais}")
print(f"Quantidade de espécies após filtragem: {especies_finais}")
print(f"Quantidade de espécies removidas: {len(especies_removidas)}")

if especies_removidas:
    print("\nEspécies sem anotações após a filtragem:")
    for especie in sorted(especies_removidas):
        print(f"- {especie}")

# ============================================================
# 6. SALVAR RESULTADO
# ============================================================

saida = "anotacoes_p10_p90.csv"
df_final.to_csv(saida, index=False)

print(f"\nArquivo salvo em: {os.path.abspath(saida)}")
