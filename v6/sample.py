import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Ler a planilha
df = pd.read_csv(os.getenv("CSV_PATH"))  # ou pd.read_csv("anotacoes.csv")

# Contar quantas anotações cada pesquisador fez por espécie
contagem = (
    df.groupby(["roi_label", "roi_user"])
      .size()
      .reset_index(name="n_anotacoes")
)

# Para cada espécie, encontrar o pesquisador com mais anotações
melhores = (
    contagem.sort_values("n_anotacoes", ascending=False)
             .drop_duplicates(subset=["roi_label"])
)


# ---------------- DEBUG ----------------
print("\n===== RESUMO POR ESPÉCIE =====\n")

for _, row in melhores.iterrows():
    especie = row["roi_label"]
    pesquisador = row["roi_user"]
    quantidade = row["n_anotacoes"]

    print(
        f"Espécie: {especie} | "
        f"Pesquisador selecionado: {pesquisador} | "
        f"Quantidade: {quantidade}"
    )

print("\n================================\n")

# Manter apenas os registros dessas combinações espécie + pesquisador
df_filtrado = df.merge(
    melhores[["roi_label", "roi_user"]],
    on=["roi_label", "roi_user"],
    how="inner"
)

# Salvar resultado
df_filtrado.to_csv("anotacoes_filtradas.csv", index=False)

print(f"Total original: {len(df)}")
print(f"Total filtrado: {len(df_filtrado)}")