import pandas as pd
from collections import Counter

# =========================
# CONFIGURAÇÃO
# =========================

arquivos = [
    "/home/a_matheusteixeira/v7/test_audio_metrics.csv",
    "/home/a_matheusteixeira/v8/test_audio_metrics.csv",
    "/home/a_matheusteixeira/v9/test_audio_metrics.csv",
    "/home/a_matheusteixeira/v10/test_audio_metrics.csv",
]

COLUNA = "audio_id"

# =========================
# LEITURA E CONTAGEM
# =========================

contagem = Counter()
planilhas_por_audio = {}

for arquivo in arquivos:
    # Lê Excel ou CSV
    if arquivo.lower().endswith(".csv"):
        df = pd.read_csv(arquivo)
    else:
        df = pd.read_excel(arquivo)

    if COLUNA not in df.columns:
        raise ValueError(
            f"A coluna '{COLUNA}' não foi encontrada em {arquivo}"
        )

    # Remove valores vazios e duplicatas dentro da planilha
    audio_ids = df[COLUNA].dropna().unique()

    for audio_id in audio_ids:
        contagem[audio_id] += 1
        planilhas_por_audio.setdefault(audio_id, []).append(arquivo)

# =========================
# RESULTADO
# =========================

if not contagem:
    print("Nenhum audio_id encontrado.")
else:
    maior_presenca = max(contagem.values())

    vencedores = [
        audio_id
        for audio_id, quantidade in contagem.items()
        if quantidade == maior_presenca
    ]

    print(f"Maior quantidade de presenças: {maior_presenca}/4")
    print(f"Quantidade de audio_id empatados: {len(vencedores)}")

    for audio_id in vencedores:
        print(f"\naudio_id: {audio_id}")
        print("Encontrado nas planilhas:")
        for planilha in planilhas_por_audio[audio_id]:
            print(f"  - {planilha}")
