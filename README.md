# Visual Behavior Analysis — Multi-Pessoa

Sistema modular de análise de comportamento visual a partir de webcam convencional ou vídeos gravados.
Detecta e rastreia múltiplas pessoas simultaneamente, extraindo em tempo real métricas de piscadas, direção do olhar, pose da cabeça, emoções, fadiga ocular, microexpressões e assimetria facial. Ao final da sessão, gera um relatório PDF completo com métricas e gráficos individuais por pessoa.

---

## Índice

1. [Requisitos de Sistema](#requisitos-de-sistema)
2. [Instalação](#instalação)
   - [Linux (Ubuntu / Debian)](#linux-ubuntu--debian)
   - [macOS](#macos)
   - [Windows](#windows)
3. [Baixar o Modelo MediaPipe](#baixar-o-modelo-mediapipe)
4. [Execução](#execução)
5. [Geração do Relatório Técnico](#geração-do-relatório-técnico)
6. [Saídas Geradas](#saídas-geradas)
7. [Configuração](#configuração-configconfigyaml)
8. [Estrutura do Projeto](#estrutura-do-projeto)
9. [Funcionalidades](#funcionalidades)
10. [Cobertura dos Atributos do Desafio](#cobertura-dos-atributos-do-desafio)
11. [Alinhamento ao Desafio Técnico](#alinhamento-ao-desafio-técnico)
12. [Decisões Técnicas](#decisões-técnicas)

---

## Requisitos de Sistema

| Item | Requisito |
|---|---|
| Python | 3.12 ou superior |
| RAM | ~500 MB (pipeline base) |
| GPU | **Não necessária** — roda inteiramente em CPU |
| Webcam | Qualquer câmera compatível com OpenCV (USB, integrada) |
| SO | Linux (Ubuntu 20.04+), macOS (12+), Windows 10/11 |

---

## Instalação

### Linux (Ubuntu / Debian)

#### 1. Instalar dependências do sistema

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    v4l-utils
```

> `libgl1`, `libglib2.0-0` e afins são necessários para o OpenCV funcionar sem interface gráfica.
> `v4l-utils` permite verificar webcams disponíveis com `v4l2-ctl --list-devices`.

#### 2. Clonar o repositório e entrar na pasta

```bash
git clone https://github.com/AIWASS23/video_behavior_mult.git
cd visual_behavior_mult
```

#### 3. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 4. Instalar dependências Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### macOS

#### 1. Instalar o Homebrew (se ainda não tiver)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Instalar Python 3.12+

```bash
brew install python@3.12
```

> Verifique a versão: `python3 --version`

#### 3. Clonar o repositório e entrar na pasta

```bash
git clone https://github.com/AIWASS23/video_behavior_mult.git
cd visual_behavior_mult
```

#### 4. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 5. Instalar dependências Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** o MediaPipe distribui wheels nativos para `arm64`. O comando acima já instala a versão correta automaticamente via pip.

> **Permissão de câmera:** na primeira execução, o macOS solicitará permissão de acesso à câmera. Aceite em **Preferências do Sistema → Privacidade e Segurança → Câmera**.

---

### Windows

#### 1. Instalar Python 3.12+

Baixe o instalador em [python.org/downloads](https://www.python.org/downloads/).

Durante a instalação, marque obrigatoriamente:
- ✅ **Add Python to PATH**
- ✅ **Install pip**

Verifique no terminal (PowerShell ou CMD):
```cmd
python --version
pip --version
```

#### 2. Instalar Git (opcional, para clonar)

Baixe em [git-scm.com](https://git-scm.com/download/win) ou extraia o ZIP do projeto manualmente.

#### 3. Abrir o terminal na pasta do projeto

```cmd
cd C:\caminho\para\visual-behavior-multi
```

#### 4. Criar e ativar o ambiente virtual

```cmd
python -m venv venv
venv\Scripts\activate
```

> No PowerShell, se aparecer erro de política de execução:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 5. Instalar dependências Python

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

> **DirectShow / câmera:** o OpenCV no Windows usa DirectShow por padrão. Se a webcam não for detectada, tente `--source 1` ou `--source 2` para outras câmeras disponíveis.

> **Exibição da janela:** no Windows, a janela OpenCV (`cv2.imshow`) funciona nativamente. Não é necessário nenhum pacote adicional.

---

## Baixar o Modelo MediaPipe

Após instalar as dependências, baixe o modelo de detecção facial (necessário uma única vez):

```bash
python download_models.py
```

O arquivo `face_landmarker.task` (~29 MB) será salvo em `models/`. Sem ele, o sistema não inicializa.

> **Sem acesso à internet?** Baixe manualmente o modelo em:
> `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
> e salve em `models/face_landmarker.task`.

---

## Execução

### Webcam ao vivo (padrão — índice 0)

```bash
python main.py
```

### Webcam alternativa (índice 1, 2…)

```bash
python main.py --source 1
```

### Arquivo de vídeo pré-gravado

```bash
python main.py --source caminho/para/video.mp4
```

### Captura por tempo determinado (sem janela)

```bash
python main.py --no-display --duration 60
```

### Todas as opções

```
python main.py [--source SOURCE] [--output DIR] [--config PATH]
               [--no-display] [--no-save] [--duration N]

  --source      Fonte de vídeo: 0 (webcam padrão), 1, 2... ou caminho de arquivo
  --output      Pasta de saída (padrão: outputs/)
  --config      Arquivo de configuração YAML (padrão: config/config.yaml)
  --no-display  Desativa a janela OpenCV (útil em servidores headless ou WSL)
  --no-save     Não salva o vídeo anotado (CSV e PDF ainda são gerados)
  --duration    Encerra automaticamente após N segundos
```

Pressione **`q`** na janela de visualização ou **`Ctrl-C`** no terminal para encerrar.

> **Linux sem interface gráfica (headless/servidor):** use sempre `--no-display`. Para exibir a janela remotamente via SSH, configure `DISPLAY` ou use `--no-display` com análise de arquivo de vídeo.

> **WSL (Windows Subsystem for Linux):** a janela OpenCV pode não funcionar dependendo da versão do WSL. Use `--no-display` ou execute diretamente no PowerShell/CMD.

---

## Saídas Geradas

Tudo salvo em `outputs/` (ou diretório indicado via `--output`):

| Arquivo | Conteúdo |
|---|---|
| `analysis_YYYYMMDD_HHMMSS.mp4` | Vídeo original com HUD multi-pessoa anotado |
| `metrics_YYYYMMDD_HHMMSS.csv` | Registro frame-a-frame de todos os atributos por pessoa |
| `report_YYYYMMDD_HHMMSS.pdf` | Relatório de sessão: sumário global, comparativo e páginas individuais por pessoa |

### Estrutura do relatório de sessão (PDF)

- **Página 1** — Capa / sumário global da sessão
- **Página 2** — Tabela comparativa entre pessoas (apenas se > 1 pessoa detectada)
- **Por pessoa** (8 páginas cada):
  - Sumário individual com foto do primeiro frame detectado e métricas agregadas
  - Gráfico 1: EAR e eventos de piscada
  - Gráfico 2: Piscadas (pizza) + Direção do olhar (barras)
  - Gráfico 3: Mapa de calor do olhar + Pose de cabeça
  - Gráfico 4: Distribuição de emoções + Linha do tempo
  - Gráfico 5: Sinal respiratório + Taxa respiratória
  - Gráfico 6: Score de fadiga + PERCLOS
  - Gráfico 7: Microexpressões + Assimetria facial

---

## Como Interpretar os Gráficos do Relatório

### Página de Sumário Individual

Exibe a foto do primeiro frame em que a pessoa foi detectada (crop do rosto com 30% de margem) ao lado de uma tabela com todas as métricas agregadas da sessão: total de frames, estabilidade de detecção, piscadas, taxa de atenção, emoção dominante, score de fadiga, total de microexpressões e assimetria média.

**O que observar:** estabilidade abaixo de 80% indica que o rosto ficou fora de enquadramento ou com oclusão frequente — as métricas desta pessoa podem ser menos confiáveis.

---

### Gráfico 1 — EAR e Eventos de Piscada

**O que mostra:** a curva do Eye Aspect Ratio (EAR) ao longo do tempo. Cada queda abrupta abaixo da linha tracejada vermelha (limiar = 0,22) representa um olho fechado. Marcadores triangulares indicam piscadas confirmadas (≥ 2 frames fechados consecutivos).

**Como interpretar:**
- **EAR estável entre 0,25 e 0,45** → olhos abertos em repouso normal.
- **Quedas frequentes e profundas** → taxa de piscadas elevada; pode indicar fadiga ocular, irritação ou resposta emocional.
- **Declínio gradual da linha base do EAR ao longo do tempo** → fechamento progressivo dos olhos, sinal clássico de sonolência.
- **Piscadas de longa duração** (marcadores maiores) → classificadas como voluntárias (> 150 ms); piscadas rápidas são reflexos involuntários.

---

### Gráfico 2 — Piscadas (pizza) + Direção do Olhar (barras)

**O que mostra:** à esquerda, a proporção entre piscadas voluntárias e involuntárias. À direita, o total de frames em cada direção do olhar (centro, esquerda, direita, cima, baixo).

**Como interpretar:**
- **Dominância de piscadas involuntárias** (curta duração) → padrão normal de vigília. Alta proporção de voluntárias pode indicar tentativa consciente de focar ou aliviar desconforto ocular.
- **Taxa total de piscadas/min:**
  - 10–20/min → faixa normal em adultos em repouso.
  - > 30/min → associado a fadiga, estresse ou irritação.
  - < 8/min → concentração intensa ou leitura (redução conhecida do reflexo palpebral).
- **Direção do olhar:** "center" dominante indica atenção à frente (câmera/tela). Alto volume de "left" ou "right" pode indicar distração frequente. "up" pode indicar busca visual ou reflexo cognitivo.

---

### Gráfico 3 — Mapa de Calor do Olhar + Pose de Cabeça

**O que mostra:** à esquerda, um scatter plot da posição normalizada da íris (eixo X = horizontal, eixo Y = vertical) com o percentual de atenção à tela. À direita, os três ângulos de Euler da cabeça ao longo do tempo: yaw (giro horizontal), pitch (inclinação vertical) e roll (rotação lateral).

**Como interpretar — Mapa de Olhar:**
- **Concentração central** → atenção focada à frente.
- **Dispersão lateral ampla** → desvios frequentes, possível distração ou varredura visual ativa.
- **`Atenção à tela`** (percentual no título): frames com olhar classificado como "center". Valores abaixo de 60% em sessões longas podem indicar dificuldade de manutenção de atenção.

**Como interpretar — Pose de Cabeça:**
- **Yaw próximo de 0°** → cabeça frontal. Yaw crescente ou oscilatório → desvios laterais frequentes.
- **Pitch:** valores positivos indicam cabeça inclinada para baixo (leitura, sonolência); negativos indicam cabeça levantada.
- **Roll:** inclinação lateral da cabeça. Variações bruscas são flagradas como `sudden_movement`.
- **Linhas muito oscilatórias** em todos os ângulos → movimentação excessiva da cabeça durante a sessão.

---

### Gráfico 4 — Distribuição de Emoções + Linha do Tempo

**O que mostra:** à esquerda, um gráfico de barras horizontais com a contagem de frames por emoção dominante (7 classes: neutro, alegria, raiva, tristeza, surpresa, medo, nojo). À direita, um scatter plot mostrando a sequência de emoções dominantes frame a frame ao longo do tempo.

**Como interpretar:**
- **Dominância de "neutral"** → padrão esperado em condições de repouso ou tarefas cognitivas neutras.
- **Picos isolados de "surprise" ou "fear"** → podem corresponder a microexpressões ou reações pontuais a estímulos.
- **"angry" persistente** → pode indicar tensão muscular facial crônica (não necessariamente raiva — rostos em repouso tensos ativam blendshapes de frown/browDown).
- **Linha do tempo com alternâncias frequentes** → expressividade emocional alta ou iluminação instável gerando ruído nos blendshapes.

---

### Gráfico 5 — Sinal Respiratório + Taxa Respiratória

**O que mostra:** à esquerda, a variação normalizada da largura nasal (proxy da respiração) ao longo do tempo — cada ciclo de subida e descida corresponde a um ciclo respiratório. À direita, a taxa respiratória estimada em respirações por minuto por janelas de 30 segundos.

**Como interpretar:**
- **Taxa normal em adultos em repouso:** 12–20 respirações/min.
- **Taxa < 12/min (bradipneia)** → respiração lenta; pode ocorrer em estados de relaxamento profundo ou sonolência.
- **Taxa > 20/min (taquipneia)** → respiração acelerada; associada a ansiedade, esforço físico ou estresse.
- **Amplitude do sinal:** ciclos com amplitude alta indicam respiração mais profunda; amplitude baixa ("shallow") indica respiração superficial — comum em estados de tensão ou fadiga.
- **Sinal irregular** → variabilidade respiratória alta. Pode ser ruído postural (movimento lateral da cabeça) ou respiração realmente irregular.
- **Limitação importante:** este é um proxy indireto baseado em landmarks nasais. Movimentos bruscos de cabeça podem gerar falsos ciclos.

---

### Gráfico 6 — Score de Fadiga + PERCLOS

**O que mostra:** à esquerda, o score de fadiga composto [0–1] ao longo do tempo, com linhas de referência para os quatro níveis (alert / mild / moderate / severe) e a curva do PERCLOS sobreposta. À direita, o PERCLOS individual em janelas de 60 segundos.

**Como interpretar:**
- **Score < 0,20 (alert)** → estado de alerta normal.
- **Score 0,20–0,40 (mild)** → sinais leves de fadiga; taxa de piscadas elevada ou leve declínio do EAR.
- **Score 0,40–0,60 (moderate)** → fadiga moderada; PERCLOS acima de 15% ou tendência de fechamento progressivo do EAR.
- **Score > 0,60 (severe)** → sonolência significativa. PERCLOS alto (olhos fechados > 60% do tempo na janela de 60 s) — nível de risco em contextos de direção veicular (padrão NHTSA).
- **Score crescente ao longo da sessão** → fadiga acumulada. Score estável indica estado de alerta mantido.
- **PERCLOS:** interpretado como a fração de tempo com olho ≥ 80% fechado. Valores acima de 0,35 por períodos prolongados são considerados críticos pelo padrão NHTSA (1998).

---

### Gráfico 7 — Microexpressões + Assimetria Facial

**O que mostra:** à esquerda, os eventos de microexpressão detectados ao longo do tempo: tipo de emoção (cor), intensidade (tamanho do marcador) e momento de ocorrência. À direita, o score de assimetria facial [0–1] ao longo do tempo, com a linha de referência para assimetria pronunciada.

**Como interpretar — Microexpressões:**
- **Eventos isolados e breves** (≤ 200 ms / ≤ 6 frames) → microexpressões genuínas, indicativas de reações emocionais espontâneas e involuntárias não sustentadas.
- **Tipo de emoção:** microexpressões de "surprise" ou "fear" são as mais comuns em contextos de reação a estímulos inesperados. "contempt" ou "disgust" podem indicar reações negativas suprimidas.
- **Alta densidade de eventos** → pode indicar expressividade emocional elevada ou, alternativamente, ruído nos blendshapes por iluminação instável ou movimentos de cabeça.
- **Intensidade (Δscore):** valores próximos de 0,15 (limiar mínimo) são microexpressões fracas; valores acima de 0,30 são expressões mais intensas.

**Como interpretar — Assimetria Facial:**
- **Score < 0,10** → rosto simétrico (faixa normal).
- **Score 0,10–0,30** → assimetria leve, dentro da variabilidade natural entre os lados do rosto.
- **Score > 0,40** → assimetria pronunciada; pode indicar tensão muscular unilateral crônica, expressões de desconforto ou, em contextos clínicos, relevante para avaliação neurológica.
- **Score estável ao longo do tempo** → padrão de simetria consistente. Picos pontuais correspondem a expressões assimétricas passageiras (piscar de um olho, sorrir de um lado).
- **Lado dominante:** o lado com blendshapes consistentemente mais ativos é reportado como "left" ou "right". "symmetric" indica equilíbrio entre os lados.

### Colunas do CSV

`timestamp`, `frame_idx`, `person_id`, `face_detected`, `blink_occurred`, `blink_type`, `ear`,
`gaze_direction`, `gaze_h`, `gaze_v`, `on_screen`, `head_yaw`, `head_pitch`, `head_roll`,
`head_frontal`, `sudden_movement`, `dominant_emotion`, `emotion_scores`, `respiratory_rate`,
`breath_depth`, `breath_phase`, `breath_amplitude`, `breath_regularity`, `nostril_width_norm`,
`fatigue_level`, `fatigue_score`, `perclos`, `ear_trend`, `microexp_detected`, `microexp_emotion`,
`microexp_intensity`, `microexp_duration_ms`, `asymmetry_score`, `asymmetry_dominant_side`,
`asymmetry_region_scores`

---

## Configuração (`config/config.yaml`)

```yaml
blink:
  ear_threshold: 0.22       # EAR abaixo deste valor → olho fechado
  min_frames: 2             # mínimo de frames fechados para registrar piscada
  voluntary_min_ms: 150     # duração mínima para classificar como voluntária

gaze:
  horizontal_threshold: 0.35
  vertical_threshold: 0.35

head_pose:
  frontal_threshold: 20.0
  sudden_movement_threshold: 15.0

emotion:
  enabled: true
  sample_interval: 5        # roda a cada N frames

fatigue:
  perclos_window_s: 60      # janela de tempo para cálculo do PERCLOS
  ear_trend_window_s: 30    # janela para regressão linear da EAR
  closed_threshold: 0.80    # score de blendshape eyeBlink para considerar olho fechado
```

---

## Estrutura do Projeto

```
visual-behavior-multi/
├── main.py                             # Ponto de entrada — pipeline completo
├── requirements.txt                    # Dependências Python
├── download_models.py                  # Baixa o modelo face_landmarker.task
├── config/
│   └── config.yaml                     # Parâmetros ajustáveis
├── models/
│   └── face_landmarker.task            # Modelo MediaPipe (~29 MB, baixado via download_models.py)
├── src/
│   ├── face_tracker.py                 # Wrapper MediaPipe Face Landmarker (Tasks API)
│   ├── person_tracker.py               # Rastreamento de IDs por centróide entre frames
│   ├── person_manager.py               # Instâncias de detectores isoladas por pessoa
│   ├── blink_detector.py               # Detecção de piscadas via EAR
│   ├── gaze_estimator.py               # Estimativa de direção do olhar via íris
│   ├── head_pose_estimator.py          # Pose 3D via solvePnP
│   ├── emotion_detector.py             # Emoções via blendshapes ARKit
│   ├── breathing_analyzer.py           # Análise de respiração via largura nasal
│   ├── fatigue_detector.py             # Fadiga ocular via PERCLOS + EAR trend
│   ├── microexpression_detector.py     # Microexpressões via spike detection
│   ├── asymmetry_detector.py           # Assimetria facial bilateral
│   ├── metrics_collector.py            # Coleta e resumo de métricas por pessoa
│   └── visualizer.py                   # HUD ao vivo + geração de relatório PDF
└── outputs/                            # Vídeo anotado, CSV e PDFs (gerados em execução)
```

---

## Funcionalidades

| Módulo | Atributos extraídos |
|---|---|
| **BlinkDetector** | Total, taxa (piscadas/min), classificação voluntária/involuntária, duração média, EAR |
| **GazeEstimator** | Direção (esq/dir/cima/baixo/centro), posição normalizada da íris, % tempo na tela |
| **HeadPoseEstimator** | Yaw / pitch / roll (°), flag frontal, detecção de movimento brusco |
| **EmotionDetector** | Emoção dominante (7 classes), distribuição temporal por frame |
| **BreathingAnalyzer** | Taxa respiratória (/min), profundidade, regularidade, fase (inspiração/expiração) |
| **FatigueDetector** | Score [0–1], nível (alert/mild/moderate/severe), PERCLOS, tendência EAR |
| **MicroexpressionDetector** | Evento detectado, tipo de emoção, intensidade, timestamp |
| **AsymmetryDetector** | Score global [0–1], lado dominante, scores por região facial |
| **MetricsCollector** | CSV frame-a-frame, resumo global e por pessoa |
| **Visualizer** | HUD multi-pessoa em tempo real + relatório PDF com gráficos individuais |

---

## Cobertura dos Atributos do Desafio

### Atributos sugeridos vs. implementação

| Atributo sugerido | Status | Módulo |
|---|---|---|
| Quantidade de piscadas | **Implementado** | `BlinkDetector` — total, taxa/min |
| Piscadas voluntárias e involuntárias | **Implementado** | `BlinkDetector` — classificação por duração (< 150 ms = involuntária) |
| Direção do olhar (cima/baixo/esq/dir) | **Implementado** | `GazeEstimator` — 5 direções + posição normalizada da íris |
| Estimativa de gaze e mapas de saliência | **Parcial** | `GazeEstimator` — direção estimada; mapa de dispersão no relatório PDF |
| Tempo olhando para determinadas regiões | **Implementado** | `GazeEstimator` — distribuição acumulada por direção |
| Tempo olhando para a tela vs. fora | **Implementado** | `GazeEstimator` — métrica `on_screen_pct` |
| Frequência de desvios de atenção | **Implementado** | `GazeEstimator` + `HeadPoseEstimator` — flag `sudden_movement` |
| Emoções predominantes ao longo do tempo | **Implementado** | `EmotionDetector` — 7 emoções via blendshapes ARKit |
| Confiança de autenticação biométrica facial | **Parcial** | `FaceTracker` — confiança por frame; `PersonTracker` — estabilidade de ID |
| Estabilidade da detecção facial | **Implementado** | `MetricsCollector` — `stability_pct` |
| Movimentação facial e mudanças bruscas | **Implementado** | `HeadPoseEstimator` — yaw/pitch/roll + flag `sudden_movement` |
| Métricas temporais, gráficos e estatísticas | **Implementado** | `MetricsCollector` (CSV) + `Visualizer` (PDF) |

### Além do que foi sugerido

| Atributo extra | Módulo |
|---|---|
| Fadiga ocular (PERCLOS + EAR trend) | `FatigueDetector` |
| Microexpressões faciais (~400 ms) | `MicroexpressionDetector` |
| Assimetria facial bilateral | `AsymmetryDetector` |
| Análise de respiração por landmarks nasais | `BreathingAnalyzer` |
| Suporte a múltiplas pessoas simultâneas | `PersonTracker` + `PersonManager` |
| Relatório PDF individual por pessoa com foto do primeiro frame | `Visualizer` |

> **Nota sobre gaze point-of-regard:** a estimativa do ponto exato na tela não é implementada — isso exige calibração com alvos visuais conhecidos, inviável com webcam convencional sem cooperação do usuário. Direção e dispersão do olhar representam o estado da arte para webcam sem calibração.

---

## Alinhamento ao Desafio Técnico

### Propor ou adaptar da literatura uma solução de visão computacional

| Técnica | Origem |
|---|---|
| **PERCLOS** | NHTSA (1998), padrão para detecção de sonolência ao volante |
| **EAR** — Eye Aspect Ratio | Soukupová & Čech, CVWW 2016 |
| **PnP solver** para pose 3D | Levenberg-Marquardt via OpenCV `solvePnP` |
| **Blendshapes ARKit** | Apple ARKit / MediaPipe Face Landmarker |
| **Microexpressões** por spike detection | Ekman & Friesen (1969, 1978) |
| **Centroid tracker** | Técnica clássica de rastreamento multi-objeto por proximidade |

### Definir estratégias de processamento

| Problema | Estratégia adotada |
|---|---|
| Identidade entre frames | Centroid tracker greedy — distância euclidiana normalizada do nariz (landmark 1) |
| Múltiplas pessoas | `PersonManager` — instâncias de detectores isoladas por `person_id` |
| Fadiga ocular | PERCLOS (60 s) + regressão linear da EAR (30 s) + blendshape `eyeWide` |
| Microexpressões | Spike detection: Δscore ≥ 0,15 e duração ≤ 6 frames em janela de 12 frames |
| Assimetria facial | 10 pares bilaterais — `\|esq − dir\|` normalizado por 0,30 |
| Respiração | Variação da largura nasal (landmarks 64 e 294) |
| Emoções sem GPU | Mapeamento de blendshapes ARKit → 7 grupos sem TensorFlow |
| Pose sem calibração | Modelo pinhole com f = largura do frame |

### Interpretar limitações

- **Centroid tracker**: IDs podem ser trocados momentaneamente quando pessoas cruzam muito próximas.
- **Respiração**: proxy indireto — ruído postural pode gerar artefatos.
- **Microexpressões**: threshold fixo pode gerar falsos positivos em iluminação instável.
- **Pose 3D**: modelo genérico de crânio sem calibração formal introduz viés angular absoluto.
- **Gaze**: direção estimada, não ponto exato na tela.
- **MediaPipe em modo IMAGE**: sem estado temporal interno entre frames.

---

## Decisões Técnicas

### MediaPipe Face Landmarker (Tasks API v0.10+)
Fornece 478 landmarks por rosto (incluindo íris) e 52 blendshapes ARKit. Roda inteiramente em CPU com latência adequada para tempo real. Suporta até 10 rostos simultâneos por configuração.

### Rastreamento de identidade — Centroid Tracker
Associa IDs persistentes por menor distância euclidiana do centróide (ponta do nariz) em coordenadas normalizadas. Tracks ausentes por mais de 30 frames são descartados.

### Fadiga — PERCLOS
Padrão NHTSA: porcentagem de frames com olho ≥ 80% fechado em janela de 60 s, complementado por tendência linear da EAR e blendshape `eyeWide`.

### Microexpressões — Spike Detection
Janela deslizante de 12 frames (~400 ms a 30 fps). Pico detectado quando Δscore ≥ 0,15 e duração ≤ 6 frames — baseado nos critérios de duração de Ekman & Friesen.

### Assimetria — Blendshapes Bilaterais
10 pares de blendshapes: `|score_esquerdo − score_direito|` normalizado por 0,30. Score próximo de 1 indica assimetria pronunciada.

### Importante — Conflito TensorFlow / MediaPipe
O MediaPipe 0.10+ usa um runtime TFLite interno incompatível com TensorFlow instalado no mesmo ambiente. **Não instale TensorFlow junto com este projeto** — causará `segfault` na inicialização.
