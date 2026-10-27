import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. LEITURA DOS DADOS
# ============================================================

df = pd.read_csv(os.getenv("CSV_PATH"))

coluna_especie = "roi_label"
coluna_valor = "roi_duration"

# Quantidades originais
total_original = len(df)
especies_originais = df[coluna_especie].nunique()

# ============================================================
# 2. CALCULAR P10 E P90 POR ESPÉCIE
# ============================================================

percentis = (
    df.groupby(coluna_especie)[coluna_valor]
      .quantile([0.10, 0.90])
      .unstack()
      .rename(columns={0.10: "P10", 0.90: "P90"})
)

# Juntar os percentis ao dataframe original
df_com_percentis = df.merge(
    percentis,
    left_on=coluna_especie,
    right_index=True,
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

# Espécies removidas durante as duas etapas
especies_originais_set = set(df[coluna_especie].dropna().unique())
especies_finais_set = set(df_final[coluna_especie].dropna().unique())

especies_removidas = especies_originais_set - especies_finais_set

# Espécies removidas especificamente por terem menos de 5 anotações
especies_menos_cinco = contagem_especies[
    contagem_especies < 5
].index.tolist()

# ============================================================
# 4. DEBUG POR ESPÉCIE
# ============================================================

print("\n===== PERCENTIS E CONTAGEM POR ESPÉCIE =====\n")

for especie, row in percentis.iterrows():
    n_original = (df[coluna_especie] == especie).sum()
    n_apos_percentis = (
        df_filtrado[coluna_especie] == especie
    ).sum()
    n_final = (df_final[coluna_especie] == especie).sum()

    motivo = ""
    if n_apos_percentis == 0:
        motivo = " | Sem anotações após P10-P90"
    elif 0 < n_apos_percentis < 5:
        motivo = " | Removida: menos de 5 anotações"

    print(
        f"{str(especie):30} "
        f"P10={row['P10']:.3f} "
        f"P90={row['P90']:.3f} "
        f"Original={n_original:5d} "
        f"Após P10-P90={n_apos_percentis:5d} "
        f"Final={n_final:5d}"
        f"{motivo}"
    )

# ============================================================
# 5. RESUMO GERAL
# ============================================================

print("\n===== RESUMO DA FILTRAGEM =====")

print(f"Total original de anotações: {len(df)}")
print(f"Total de anotações filtradas: {len(df_final)}")

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
