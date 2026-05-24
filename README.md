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
  - EAR e eventos de piscada ao longo do tempo
  - Distribuição e mapa de direção do olhar
  - Pose de cabeça (yaw / pitch / roll)
  - Distribuição de emoções
  - Análise de respiração
  - Fadiga ocular (PERCLOS + score)
  - Microexpressões e assimetria facial

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
