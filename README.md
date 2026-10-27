# Segmentação automática de vocalizações de aves com YOLO

Código do trabalho de iniciação científica em Ciência da Computação (UTFPR, Câmpus Campo Mourão) sobre **localização espaço-temporal de vocalizações de aves** em gravações de monitoramento acústico passivo (PAM). Os áudios são convertidos em espectrogramas na escala Mel e tratados como imagens; o **YOLO** prevê caixas delimitadoras (tempo × frequência) para cada vocalização.

> Autor: Matheus de Oliveira Teixeira · Orientador: Prof. Dr. Juliano Henrique Foleis

## Visão geral

```
áudios (WAV, ~60 s) + anotações (CSV)
   → reamostragem 32.768 Hz → [subtração espectral] → STFT (2.048) → escala Mel (128 bandas)
   → espectrogramas + caixas no formato YOLO
   → treino do YOLO (v8n / v26m) → inferência → métricas (IoU, DSC, precisão, recall, F1) → figuras de erro
```

Cada pasta `vN/` é **uma abordagem experimental** (um conjunto de scripts e um `.env` próprio). Os scripts de cada pasta seguem o mesmo fluxo; as pastas se diferenciam pelo tratamento dos dados.

## Estrutura do repositório

```
scripts/
├── v1/ … v10/            # Blocos 1 e 2: partição 70/20/10 (ver tabela abaixo)
├── v11/ … v28/           # Bloco 3: validação cruzada de 5 folds, teste fixo de 20 áudios
├── anotations.py         # desenha as anotações de uma espécie (ex.: *Patagioenas picazuro*) sobre o espectrograma
├── histogram.py          # distribuição de anotações por espécie
├── histogramuser.py      # distribuição de anotações por espécie para cada anotador
├── histogramuser2.py     # quantidade de anotações por anotador
├── status.py             # estatísticas gerais do dataset (classes, anotadores, durações)
├── spectrogram.py        # espectrograma linear e Mel de um áudio de exemplo
├── verifyv6v10.py        # conta em quantas planilhas de teste (v6–v10) cada áudio aparece
├── verifyv7v10.py        # idem, para v7–v10
├── gerar_graficos_cv.py            # gráficos por grupo (média ± desvio, 5 métricas)
├── gerar_graficos_gerais.py        # gráficos comparando as 18 abordagens do Bloco 3
└── gerar_graficos_reorganizados.py # média e melhor fold de precisão/recall/F1
```

## Mapeamento versão → abordagem

> Mapeamento inferido da leitura dos scripts e confirmado pelo autor para `v1` e `v5`.

### Blocos 1 e 2 (partição 70/20/10)

| Pasta | Abordagem | Observação |
|---|---|---|
| `v1` | Pipeline inicial (espectrograma integral de 60 s, imagem 640×640) | código antigo com validação cruzada e teste aleatório de 20%; foi modificado por cima do original ao iniciar a versão com validação cruzada, então **não reproduz exatamente a execução da Base do Bloco 1** |
| `v2` | Janelamento temporal | janelas de 10 s, sobreposição de 50%, imagem 512×512 |
| `v3` | Redução de ruído (subtração espectral) | janelas de 10 s |
| `v4/low-fr`, `v4/high-fr` | Divisão em dois modelos por banda de frequência | 0–3.072 Hz e 2.048–16.384 Hz |
| `v5` | Diferença espectral | experimento exploratório, **não faz parte do trabalho final** |
| `v6` | Limpeza mantendo o anotador majoritário por classe | usa `anotacoes_filtradas.csv` |
| `v7` | Percentis P10–P90 **por espécie** | `sample.py` agrupa por `roi_label` |
| `v8` | Percentis P10–P90 **por anotador** | `sample.py` agrupa por `roi_user` |
| `v9` | Percentis P25–P75 **por anotador** | |
| `v10` | Percentis P25–P75 **por espécie** | |

### Bloco 3 (validação cruzada de 5 folds, teste fixo de 20 áudios)

| Grupo | sem subtração espectral | com subtração espectral | sem subtração + ≥5 amostras | com subtração + ≥5 amostras |
|---|---|---|---|---|
| Base (sem limpeza) | `v11` | `v12` | – | – |
| Espécie P10–P90 | `v17` | `v13` | `v22` | `v21` |
| Espécie P25–P75 | `v19` | `v15` | `v26` | `v25` |
| Anotador P10–P90 | `v18` | `v14` | `v24` | `v23` |
| Anotador P25–P75 | `v20` | `v16` | `v28` | `v27` |

"≥5 amostras" indica a remoção de espécies com menos de cinco anotações.

## Fluxo dentro de cada pasta

**Blocos 1 e 2 (`v2`–`v10`)**

| Ordem | Script | Função |
|---|---|---|
| 1 | `sample.py` (v6–v10) | gera o CSV de anotações filtradas (percentis) |
| 2 | `slices.py` | gera os espectrogramas Mel e os rótulos no formato YOLO |
| 3 | `split.py` | separa treino/validação/teste sem repetir áudio entre conjuntos |
| 4 | `yolotrain.py` | treina o YOLO |
| 5 | `yoloinfere.py` | roda a inferência no conjunto de teste |
| 6 | `metricscalc.py` | calcula IoU, DSC e demais métricas |
| 7 | `verify.py` | reconstrói as predições sobre o espectrograma (TP, FP, FN) |

**Bloco 3 (`v11`–`v28`)**

| Ordem | Script | Função |
|---|---|---|
| 1 | `sample.py` | gera o CSV filtrado por percentis (ausente em `v11` e `v12`) |
| 2 | `dataset.py` | gera as imagens de 60 s (640×640) e os rótulos; aplica a subtração espectral nas versões "com ruído" |
| 3 | `splittrain.py` | separa os 20 áudios de teste fixos e treina o `yolo26m` em 5 folds (200 épocas, lote 64, AdamW, lr 0,001) |
| 4 | `testall.py` | inferência de cada fold no conjunto de teste e `test_folds_results.csv` (IoU/DSC) |
| 5 | `verifyall.py` | pareamento predição–gabarito, `fold_summary.csv` (precisão, recall, F1, TP, FP, FN) e figuras de erro |
| – | `metrics.py` | funções de IoU, DSC e pareamento usadas pelos demais |
| – | `test.py`, `verify.py` | versões de um único fold (melhor modelo) para inspeção visual |

Depois de rodar `verifyall.py` em todas as versões, os scripts `gerar_graficos_*.py` leem `reconstructions/fold_summary.csv` e `test_folds_results.csv` de cada pasta e geram os gráficos.

## Requisitos

- Python 3.10+ e GPU com CUDA para o treinamento
- Dependências em `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração

Cada pasta lê seus caminhos de um arquivo `.env` (`python-dotenv`). Copie `.env.example` para `.env` em cada pasta e ajuste:

| Variável | Descrição |
|---|---|
| `CSV_PATH` | CSV com as anotações originais (colunas usadas: `roi_label`, `roi_user`, `roi_min_freq`, `roi_max_freq`, `roi_duration`, `soundscape_path`, …) |
| `CSV_SAMPLED_PATH` | CSV filtrado gerado por `sample.py` (percentis) |
| `AUDIO_DIR` | pasta com os arquivos WAV |
| `BASE_DIR` | pasta de saída da versão (imagens, rótulos, pesos e resultados) |

Os scripts de `v1`–`v4`, `v11` e os `gerar_graficos_*.py` ainda têm caminhos escritos no código (`/home/a_matheusteixeira/...`); ajuste-os antes de rodar em outra máquina.

## Dados

Os áudios e as anotações **não estão neste repositório**. A base contém 1.541 registros de cerca de 60 s com 28.732 anotações de cinco especialistas, 172 espécies e 54 pontos de coleta (2023–2024), cedida pelo grupo de biologia da Universidade Estadual de Londrina (UEL). Para reutilizar o código, é preciso solicitar a base aos responsáveis.

## Resultados

Resumo dos resultados principais (detalhes no TCC e no relatório de estágio):

| Bloco | Abordagem | IoU | DSC | F1 |
|---|---|---|---|---|
| 1 | Base (média de 3 execuções) | 0,551 | 0,682 | – |
| 2 | P25–P75 por espécie (média de 3 execuções) | 0,616 | 0,731 | – |
| 3 | Base (validação cruzada) | 0,599 | 0,723 | 0,509 |
| 3 | P25–P75 por anotador, sem subtração espectral | 0,633 | 0,759 | 0,567 |
| 3 | P25–P75 por espécie, sem subtração espectral, ≥5 amostras | 0,672 | 0,776 | 0,293 |

Valores de blocos distintos não são diretamente comparáveis (partição dos dados, configuração de treino e janelamento diferem). O principal achado é que o tratamento das anotações influencia o desempenho mais do que as demais estratégias testadas.

## Como citar

```
TEIXEIRA, M. O. Segmentação de Vocalizações de Pássaros com YOLO: Avaliação de Técnicas de Processamento de Sinais e Limpeza de Anotações.
Trabalho de Conclusão de Curso (Bacharelado em Ciência da Computação) — UTFPR, Câmpus Campo Mourão, 2026.
```

## Licença

© 2026 Matheus de Oliveira Teixeira. Este repositório (código, documentação e figuras) é licenciado sob a [Creative Commons Atribuição-CompartilhaIgual 4.0 Internacional (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.pt-br). O texto completo está em [`LICENSE`](LICENSE).

[![CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-sa/4.0/deed.pt-br)

Os áudios e as anotações da UEL **não** são cobertos por esta licença e não fazem parte do repositório; seu uso segue as regras definidas pelos responsáveis pela base.
