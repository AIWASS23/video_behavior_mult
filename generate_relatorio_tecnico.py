#!/usr/bin/env python3
"""
Gerador do Relatório Técnico — Visual Behavior Analysis Multi-Pessoa

Uso:
    python generate_relatorio_tecnico.py
    python generate_relatorio_tecnico.py --csv outputs/metrics_XXXX.csv

Salva em: outputs/relatorio_tecnico_YYYYMMDD_HHMMSS.pdf
"""

import argparse
import glob
import json
import os
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# ── Paleta ────────────────────────────────────────────────────────────────────
PRIMARY   = "#1a1a2e"
ACCENT    = "#e94560"
SECONDARY = "#16213e"
GRAY      = "#666666"
LGRAY     = "#cccccc"
GREEN     = "#27ae60"
BLUE      = "#2471a3"
ORANGE    = "#d35400"
PURPLE    = "#8e44ad"
TEAL      = "#148f77"
GOLD      = "#d4ac0d"
LIGHT_BG  = "#f9f9f9"

PAGE_W, PAGE_H = 8.27, 11.69   # A4 portrait

PERSON_COLORS = ["#e94560", "#2471a3", "#27ae60", "#d35400",
                 "#8e44ad", "#148f77", "#d4ac0d", "#2e86c1"]

# ── Utilitários de layout ──────────────────────────────────────────────────────

def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")
    return fig


def _overlay_ax(fig):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def hline(fig, y, x0=0.08, x1=0.92, color=ACCENT, lw=1.2):
    ax = _overlay_ax(fig)
    ax.plot([x0, x1], [y, y], color=color, lw=lw)


def page_header(fig, text):
    fig.text(0.08, 0.967, text, fontsize=7.5, color=GRAY, va="top",
             fontfamily="monospace")
    hline(fig, 0.958, color=LGRAY, lw=0.6)


def page_footer(fig, num, total=""):
    hline(fig, 0.047, color=LGRAY, lw=0.6)
    label = f"Visual Behavior Analysis — Multi-Pessoa  ·  Relatório Técnico"
    if total:
        label += f"  ·  Pág. {num} / {total}"
    else:
        label += f"  ·  Pág. {num}"
    fig.text(0.5, 0.030, label, ha="center", fontsize=7.5, color=GRAY)


def section_title(fig, y, text, size=12):
    fig.text(0.08, y, text, fontsize=size, fontweight="bold",
             color=PRIMARY, va="top")
    hline(fig, y - 0.022, color=ACCENT, lw=1.2)
    return y - 0.052


def subsection(fig, y, text, size=10):
    fig.text(0.08, y, text, fontsize=size, fontweight="bold",
             color=SECONDARY, va="top")
    return y - 0.030


def body(fig, y, text, size=8.8, color="#222222", lh=1.60, wrap=108, x=0.08):
    """Renders wrapped text, returns new y below the block."""
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, wrap) or [""])
    rendered = "\n".join(lines)
    fig.text(x, y, rendered, fontsize=size, color=color,
             va="top", linespacing=lh)
    consumed = len(lines) * size * lh * 1.34 / 72.0 / PAGE_H
    return y - consumed


def bullet_list(fig, y, items, size=8.8, color="#222222", lh=1.58, x=0.10):
    lines = []
    for item in items:
        wrapped = textwrap.wrap(item, 102)
        if wrapped:
            lines.append("  •  " + wrapped[0])
            for cont in wrapped[1:]:
                lines.append("       " + cont)
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color,
             va="top", linespacing=lh)
    consumed = len(lines) * size * lh * 1.34 / 72.0 / PAGE_H
    return y - consumed


def colored_box(fig, x, y, w, h, label, sublabel="", bg=PRIMARY, fg="white", fontsize=8.5):
    ax = _overlay_ax(fig)
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.005",
                         facecolor=bg, edgecolor="white", linewidth=1.2,
                         transform=ax.transAxes, zorder=5)
    ax.add_patch(box)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w / 2, cy + 0.008, label, ha="center", va="center",
                fontsize=fontsize, color=fg, fontweight="bold", zorder=6)
        ax.text(x + w / 2, cy - 0.014, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=fg, zorder=6, alpha=0.85)
    else:
        ax.text(x + w / 2, cy, label, ha="center", va="center",
                fontsize=fontsize, color=fg, fontweight="bold", zorder=6)


def arrow(fig, x0, y0, x1, y1, color=GRAY, lw=1.2):
    ax = _overlay_ax(fig)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=4)


# ── Carregamento de dados ──────────────────────────────────────────────────────

def load_csv(csv_path=None):
    if csv_path:
        return pd.read_csv(csv_path)
    pattern = os.path.join(os.path.dirname(__file__), "outputs", "metrics_*.csv")
    files = sorted(glob.glob(pattern))
    if files:
        return pd.read_csv(files[-1])
    return None


# ── Páginas ────────────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = new_page()
    fig.patch.set_facecolor(PRIMARY)

    # Faixa superior
    ax = _overlay_ax(fig)
    ax.add_patch(FancyBboxPatch((0, 0.82), 1, 0.18,
                                boxstyle="square,pad=0",
                                facecolor=SECONDARY, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0, 0), 1, 0.10,
                                boxstyle="square,pad=0",
                                facecolor=SECONDARY, edgecolor="none"))
    ax.plot([0.08, 0.92], [0.80, 0.80], color=ACCENT, lw=2)
    ax.plot([0.08, 0.92], [0.10, 0.10], color=ACCENT, lw=1)

    fig.text(0.5, 0.92, "R  E  L  A  T  Ó  R  I  O     T  É  C  N  I  C  O", ha="center",
             fontsize=11, color=ACCENT, fontweight="bold", va="top")
    fig.text(0.5, 0.74, "Visual Behavior Analysis", ha="center",
             fontsize=28, color="white", fontweight="bold", va="top")
    fig.text(0.5, 0.67, "Multi-Pessoa", ha="center",
             fontsize=20, color=ACCENT, fontweight="bold", va="top")

    fig.text(0.5, 0.60, "Sistema de análise de comportamento visual em tempo real\n"
             "via webcam convencional com suporte a múltiplas pessoas simultâneas",
             ha="center", fontsize=10.5, color=LGRAY, va="top", linespacing=1.7)

    fig.text(0.5, 0.46,
             "Extração de atributos visuais  ·  Métricas comportamentais  ·  Rastreamento multi-pessoa\n"
             "Fadiga ocular (PERCLOS)  ·  Microexpressões  ·  Assimetria facial  ·  Pose 3D",
             ha="center", fontsize=9, color="#aaaaaa", va="top", linespacing=1.8)

    fig.text(0.5, 0.33, "Módulos implementados", ha="center",
             fontsize=8, color=ACCENT, va="top", fontweight="bold")

    modules = ["BlinkDetector", "GazeEstimator", "HeadPoseEstimator", "EmotionDetector",
               "BreathingAnalyzer", "FatigueDetector", "MicroexpressionDetector",
               "AsymmetryDetector", "PersonTracker", "PersonManager"]
    cols = 5
    bw, bh = 0.155, 0.038
    for i, m in enumerate(modules):
        col = i % cols
        row = i // cols
        bx = 0.08 + col * (bw + 0.018)
        by = 0.285 - row * (bh + 0.012)
        colored_box(fig, bx, by, bw, bh, m, bg=BLUE if i < 8 else ACCENT, fontsize=7.5)

    fig.text(0.5, 0.175, "Implementado em Python 3.10+  com  MediaPipe · OpenCV · Matplotlib · NumPy · SciPy",
             ha="center", fontsize=8.5, color="#aaaaaa", va="top")

    fig.text(0.5, 0.075, datetime.now().strftime("%B de %Y"),
             ha="center", fontsize=9, color=LGRAY, va="top")
    fig.text(0.5, 0.053, "Versão 1.0", ha="center", fontsize=8, color=GRAY, va="top")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resumo(pdf, page_num):
    fig = new_page()
    page_header(fig, "Resumo Executivo")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "Resumo Executivo")
    y = body(fig, y, (
        "Este relatório descreve o desenvolvimento de um sistema modular de análise de comportamento visual "
        "construído inteiramente em Python, capaz de processar vídeo capturado por webcam convencional — em "
        "tempo real ou a partir de arquivos pré-gravados — para extrair, quantificar e apresentar atributos "
        "visuais relacionados à atenção, estado emocional, fadiga e comportamento facial de múltiplas pessoas "
        "simultaneamente."
    ))
    y -= 0.010

    y = body(fig, y, (
        "O sistema integra dez módulos independentes, cada um responsável por uma dimensão específica do "
        "comportamento visual: detecção e rastreamento facial (MediaPipe Face Landmarker), rastreamento de "
        "identidade multi-pessoa por centróide (PersonTracker), detecção de piscadas via Eye Aspect Ratio "
        "(BlinkDetector), estimativa de direção do olhar (GazeEstimator), pose de cabeça em 3D via solvePnP "
        "(HeadPoseEstimator), emoções via blendshapes ARKit (EmotionDetector), análise de respiração por "
        "variação de largura nasal (BreathingAnalyzer), fadiga ocular por PERCLOS (FatigueDetector), "
        "microexpressões por spike detection em janela deslizante (MicroexpressionDetector) e assimetria "
        "facial bilateral (AsymmetryDetector)."
    ))
    y -= 0.010

    y = body(fig, y, (
        "Todas as métricas são registradas frame a frame em formato CSV e consolidadas em um relatório PDF "
        "ao final de cada sessão, com até oito páginas de gráficos por pessoa rastreada. O sistema opera "
        "inteiramente em CPU, sem necessidade de GPU, com latência adequada para uso em tempo real em "
        "hardware convencional."
    ))
    y -= 0.018

    y = subsection(fig, y, "Palavras-chave")
    y = body(fig, y,
             "Análise facial  ·  Comportamento visual  ·  MediaPipe  ·  EAR  ·  PERCLOS  ·  "
             "Gaze estimation  ·  Rastreamento multi-pessoa  ·  Microexpressões  ·  Fadiga ocular  ·  "
             "Assimetria facial  ·  Python  ·  OpenCV",
             color=GRAY)
    y -= 0.018

    y = subsection(fig, y, "Tecnologias principais")
    techs = [
        ("Python 3.12+", "Linguagem de programação principal"),
        ("MediaPipe Face Landmarker (Tasks API v0.10+)", "478 landmarks + 52 blendshapes ARKit por rosto"),
        ("OpenCV 4.x", "Captura de vídeo, processamento de imagem, HUD em tempo real"),
        ("NumPy / SciPy", "Álgebra linear, regressão, operações vetoriais"),
        ("Matplotlib / PdfPages", "Visualização, HUD e geração de relatórios PDF"),
        ("PyYAML", "Carregamento de configuração parametrizável"),
    ]
    for name, desc in techs:
        fig.text(0.10, y, f"•  ", fontsize=9, color=ACCENT, va="top")
        fig.text(0.13, y, name, fontsize=9, color=PRIMARY, va="top", fontweight="bold")
        fig.text(0.13, y - 0.020, f"   {desc}", fontsize=8.5, color=GRAY, va="top")
        y -= 0.042

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_introducao(pdf, page_num):
    fig = new_page()
    page_header(fig, "1. Introdução e Motivação")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "1. Introdução e Motivação")
    y = body(fig, y, (
        "A análise automática de comportamento visual humano a partir de vídeo é uma área com ampla aplicação "
        "em psicologia cognitiva, segurança veicular, interfaces homem-computador, monitoramento de atenção em "
        "ambientes educacionais e triagem clínica. Historicamente, a obtenção de métricas confiáveis exigia "
        "equipamentos especializados — rastreadores de olhar por infravermelho, câmeras de alta velocidade ou "
        "sensores de eletromiografia facial — com custo proibitivo para uso em larga escala."
    ))
    y -= 0.010

    y = body(fig, y, (
        "O avanço de modelos de visão computacional baseados em aprendizado profundo, em particular o "
        "MediaPipe Face Landmarker do Google (Lugaresi et al., 2019; Grishchenko et al., 2022), tornou possível "
        "extrair 478 marcos faciais e 52 coeficientes de mistura (blendshapes) ARKit de forma robusta e em "
        "tempo real a partir de câmeras RGB convencionais, sem hardware adicional."
    ))
    y -= 0.010

    y = body(fig, y, (
        "Este projeto surge neste contexto, propondo um sistema completo que vai da captura bruta do vídeo até "
        "a geração de métricas comportamentais interpretáveis, incluindo suporte a múltiplas pessoas "
        "simultâneas com rastreamento de identidade persistente entre frames — funcionalidade ausente na "
        "maioria das implementações baseadas em MediaPipe disponíveis na literatura."
    ))
    y -= 0.018

    y = section_title(fig, y, "2. Objetivos")
    y = bullet_list(fig, y, [
        "Implementar um pipeline completo de análise de comportamento visual a partir de webcam convencional, "
        "sem necessidade de GPU ou hardware especializado.",
        "Extrair, quantificar e apresentar atributos visuais relacionados à atenção, estado emocional, fadiga "
        "e comportamento facial com base em técnicas consolidadas da literatura.",
        "Suportar múltiplas pessoas simultaneamente com atribuição de identidade persistente entre frames "
        "via rastreamento por centróide.",
        "Detectar fadiga ocular usando o padrão PERCLOS (NHTSA, 1998) combinado com análise de tendência "
        "do Eye Aspect Ratio (EAR).",
        "Detectar microexpressões faciais com base na duração característica descrita por Ekman & Friesen (1969, 1978).",
        "Quantificar assimetria facial bilateral usando pares de blendshapes homólogos.",
        "Gerar relatórios PDF individuais por pessoa com métricas, gráficos temporais e estatísticas consolidadas.",
        "Exportar todas as métricas em formato CSV para análise posterior ou integração com sistemas externos.",
    ])
    y -= 0.018

    y = section_title(fig, y, "3. Escopo e Abordagem")
    y = body(fig, y, (
        "O sistema processa vídeo frame a frame em modo síncrono (não há paralelismo de threads nos detectores "
        "individuais). A taxa de processamento efetiva depende do hardware, variando entre 15 e 30 fps em CPU "
        "moderna para uma pessoa, com degradação proporcional ao número de rostos detectados. A entrada pode "
        "ser qualquer fonte compatível com OpenCV: webcam USB/integrada ou arquivo de vídeo (MP4, AVI, MKV)."
    ))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_fundamentacao(pdf, page_num):
    fig = new_page()
    page_header(fig, "4. Fundamentação Teórica")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "4. Fundamentação Teórica")

    topics = [
        ("4.1  Eye Aspect Ratio (EAR)",
         "Soukupová & Čech (2016) propuseram o EAR como razão entre a soma das distâncias euclidianas "
         "verticais e a distância horizontal do olho:\n"
         "    EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)\n"
         "Quando o olho se fecha, o EAR cai abruptamente para próximo de zero. O limiar de fechamento "
         "(padrão: EAR < 0,22) é parametrizável. A classificação voluntária/involuntária é baseada na "
         "duração: piscadas com duração inferior a 150 ms são classificadas como reflexos involuntários, "
         "conforme Stern et al. (1984)."),
        ("4.2  PERCLOS — Fadiga Ocular",
         "O PERCLOS (Percentage of Eye Closure) é o padrão adotado pela NHTSA (1998) para detecção de "
         "sonolência ao volante. Mede a proporção de tempo em que os olhos permanecem fechados (score "
         "de blendshape eyeBlink ≥ 0,80) em uma janela deslizante de 60 segundos. O sistema complementa "
         "o PERCLOS com uma regressão linear da EAR nos últimos 30 segundos (tendência de fechamento "
         "progressivo) e o blendshape eyeWide (compensação à sonolência), resultando em um score "
         "composto de fadiga normalizado em [0, 1]."),
        ("4.3  Estimativa de Pose de Cabeça — solvePnP",
         "A estimativa de pose 3D da cabeça utiliza o algoritmo PnP (Perspective-n-Point) com seis "
         "landmarks estáveis (nariz, queixo, cantos dos olhos e boca) e um modelo genérico de crânio "
         "em coordenadas 3D. O OpenCV resolve a correspondência 2D-3D via Levenberg-Marquardt (com "
         "RANSAC), produzindo os ângulos de Euler yaw (giro horizontal), pitch (inclinação vertical) e "
         "roll (rotação lateral). A matriz de câmera é aproximada por modelo pinhole com distância focal "
         "estimada — suficiente para webcam sem calibração formal (Gee & Cipolla, 1994)."),
        ("4.4  Blendshapes ARKit e Emoções",
         "O MediaPipe Face Landmarker produz 52 coeficientes de mistura (blendshapes) compatíveis com o "
         "padrão ARKit da Apple (Sexton, 2017). Cada coeficiente representa a intensidade de uma ação "
         "muscular facial específica (ex: eyeBlinkLeft, mouthSmileRight). O sistema mapeia esses "
         "coeficientes para as sete emoções básicas de Ekman (1992): alegria, surpresa, medo, nojo, "
         "raiva, tristeza e neutro, sem dependência de modelo adicional ou GPU."),
        ("4.5  Microexpressões Faciais",
         "Microexpressões são contrações musculares faciais involuntárias com duração entre 40 ms e "
         "500 ms, descritas por Ekman & Friesen (1969, 1978) como indicadores de estados emocionais "
         "suprimidos. O sistema implementa detecção por spike detection: janela deslizante de 12 frames "
         "(~400 ms a 30 fps), com baseline = média dos primeiros 6 frames e detecção quando "
         "Δscore ≥ 0,15 com duração ≤ 6 frames. Abordagem compatível com Li et al. (2013)."),
        ("4.6  Rastreamento Multi-Pessoa por Centróide",
         "O rastreamento de identidade utiliza o algoritmo de centróide descrito por Rosebrock (2018): "
         "para cada frame, o centróide de cada rosto detectado (ponta do nariz, landmark 1) é comparado "
         "com os centróides dos tracks ativos. A associação é feita de forma greedy pela menor distância "
         "euclidiana em coordenadas normalizadas (limiar: 0,20). Tracks ausentes por mais de 30 frames "
         "consecutivos são descartados."),
    ]

    for title, text in topics:
        y = subsection(fig, y, title)
        y = body(fig, y, text)
        y -= 0.014

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_arquitetura(pdf, page_num):
    fig = new_page()
    page_header(fig, "5. Arquitetura do Sistema")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "5. Arquitetura do Sistema")
    y = body(fig, y, (
        "O pipeline é organizado em camadas sequenciais com responsabilidades bem definidas. Cada módulo "
        "recebe dados estruturados da camada anterior e é completamente substituível sem afetar os demais. "
        "Abaixo, o diagrama completo do fluxo de processamento por frame:"
    ))
    y -= 0.015

    # Pipeline diagram
    ax = _overlay_ax(fig)

    # Input
    colored_box(fig, 0.08, 0.705, 0.84, 0.042, "ENTRADA  —  Webcam (cv2.VideoCapture) ou arquivo de vídeo (MP4 / AVI / MKV)",
                bg=SECONDARY, fontsize=8.5)
    arrow(fig, 0.50, 0.705, 0.50, 0.685)

    # Face Detection row
    colored_box(fig, 0.08, 0.648, 0.38, 0.040,
                "FaceTracker", "MediaPipe Face Landmarker\n478 landmarks + 52 blendshapes",
                bg=BLUE, fontsize=8)
    colored_box(fig, 0.54, 0.648, 0.38, 0.040,
                "PersonTracker", "Centroid tracker\nIDs persistentes entre frames",
                bg=PURPLE, fontsize=8)
    ax.annotate("", xy=(0.54, 0.668), xytext=(0.46, 0.668),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2, mutation_scale=11))
    arrow(fig, 0.50, 0.648, 0.50, 0.628)

    colored_box(fig, 0.08, 0.610, 0.84, 0.032,
                "PersonManager  —  instâncias isoladas de detectores por person_id",
                bg=TEAL, fontsize=8.5)
    arrow(fig, 0.50, 0.610, 0.50, 0.590)

    # Detectors row 1
    dets1 = [
        ("BlinkDetector", "EAR + voluntária\n/ involuntária", GREEN),
        ("GazeEstimator", "Direção do olhar\n+ on-screen %", BLUE),
        ("HeadPoseEstimator", "Yaw / Pitch / Roll\n(solvePnP)", ORANGE),
        ("EmotionDetector", "7 emoções\nblendshapes ARKit", PURPLE),
    ]
    bw = 0.195
    for i, (name, sub, color) in enumerate(dets1):
        bx = 0.08 + i * (bw + 0.015)
        colored_box(fig, bx, 0.548, bw, 0.044, name, sub, bg=color, fontsize=7.5)
    arrow(fig, 0.50, 0.548, 0.50, 0.528)

    # Detectors row 2
    dets2 = [
        ("BreathingAnalyzer", "Taxa resp.\nlargura nasal", TEAL),
        ("FatigueDetector", "PERCLOS\n+ EAR trend", ACCENT),
        ("MicroexpressionDetector", "Spike detection\n~400 ms", GOLD),
        ("AsymmetryDetector", "10 pares\nbilaterais", SECONDARY),
    ]
    for i, (name, sub, color) in enumerate(dets2):
        bx = 0.08 + i * (bw + 0.015)
        colored_box(fig, bx, 0.484, bw, 0.044, name, sub, bg=color, fontsize=7.5)
    arrow(fig, 0.50, 0.484, 0.50, 0.462)

    # Collector + Visualizer
    colored_box(fig, 0.08, 0.428, 0.38, 0.038,
                "MetricsCollector", "CSV frame-a-frame\nResumo por pessoa",
                bg=SECONDARY, fontsize=8)
    colored_box(fig, 0.54, 0.428, 0.38, 0.038,
                "Visualizer", "HUD ao vivo (OpenCV)\n+ Relatório PDF (Matplotlib)",
                bg=PRIMARY, fontsize=8)
    ax.plot([0.46, 0.54], [0.447, 0.447], color=GRAY, lw=1.2)

    # Outputs
    arrow(fig, 0.27, 0.428, 0.27, 0.408)
    arrow(fig, 0.73, 0.428, 0.73, 0.408)
    colored_box(fig, 0.08, 0.370, 0.38, 0.038,
                "metrics_YYYYMMDD.csv", bg="#2c3e50", fontsize=8.5)
    colored_box(fig, 0.54, 0.370, 0.38, 0.038,
                "report_YYYYMMDD.pdf  +  analysis_YYYYMMDD.mp4", bg="#2c3e50", fontsize=8)

    # Legend
    y = 0.330
    fig.text(0.08, y, "Legenda:", fontsize=8, color=GRAY, va="top", fontweight="bold")
    legend_items = [
        (BLUE, "Detecção facial"),
        (PURPLE, "Rastreamento de identidade"),
        (TEAL, "Gestão de instâncias"),
        (GREEN, "Detector individual"),
        (ACCENT, "Fadiga / atenção"),
        (PRIMARY, "Saída / relatório"),
    ]
    for i, (color, label) in enumerate(legend_items):
        xi = 0.08 + i * 0.145
        colored_box(fig, xi, 0.298, 0.10, 0.020, label, bg=color, fontsize=6.8)

    y = 0.270
    y = body(fig, y, (
        "A arquitetura segue o padrão Pipe-and-Filter: cada estágio transforma a entrada em uma estrutura "
        "de dados bem definida (FaceData, dict de resultados por detector) sem efeitos colaterais sobre os "
        "demais. O PersonManager implementa o padrão Registry, mantendo um dicionário de instâncias indexado "
        "por person_id e criando novas instâncias sob demanda (lazy initialization)."
    ))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_metodologia_1(pdf, page_num):
    """Detecção facial + Piscadas + Fadiga."""
    fig = new_page()
    page_header(fig, "5. Metodologia — Detecção Facial · Piscadas · Fadiga Ocular")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "5. Metodologia")

    y = subsection(fig, y, "5.1  Detecção Facial — MediaPipe Face Landmarker")
    y = body(fig, y, (
        "O módulo FaceTracker encapsula o MediaPipe Face Landmarker (Tasks API v0.10+), configurado para "
        "detectar até 10 rostos simultâneos por frame. O modelo face_landmarker.task (~29 MB) roda via "
        "TFLite inteiramente em CPU e produz, para cada rosto:"
    ))
    y = bullet_list(fig, y, [
        "478 NormalizedLandmarks: coordenadas (x, y, z) normalizadas em [0, 1] relativas às dimensões do frame. "
        "Os landmarks 468–477 correspondem às íris (esquerda e direita), habilitados por output_face_blendshapes=True.",
        "52 blendshapes ARKit: coeficientes em [0, 1] representando a intensidade de cada ação muscular facial. "
        "Exemplos: eyeBlinkLeft (0=aberto, 1=fechado), mouthSmileRight (0=neutro, 1=sorriso máximo).",
    ])
    y -= 0.006

    y = subsection(fig, y, "5.2  Rastreamento Multi-Pessoa — PersonTracker + PersonManager")
    y = body(fig, y, (
        "O PersonTracker mantém um dicionário de tracks ativos, onde cada track armazena o centróide atual "
        "(coordenadas normalizadas da ponta do nariz, landmark 1) e o contador de frames ausentes. A cada "
        "frame, o algoritmo greedy associa detecções a tracks pela menor distância euclidiana dentro do "
        "limiar de 0,20. Tracks ausentes por mais de 30 frames consecutivos são descartados. O PersonManager "
        "mantém instâncias independentes de todos os oito detectores por person_id, garantindo que o estado "
        "temporal de cada detector (histórico de EAR, janelas deslizantes) não seja compartilhado entre pessoas."
    ))
    y -= 0.010

    y = subsection(fig, y, "5.3  Detecção de Piscadas — BlinkDetector (EAR)")
    y = body(fig, y, (
        "O Eye Aspect Ratio é calculado a partir de seis landmarks por olho (cantos medial/lateral + "
        "pálpebras superior e inferior) extraídos dos 478 landmarks do MediaPipe. A fórmula:\n"
        "\n"
        "    EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 × ||p1 - p4|| )\n"
        "\n"
        "O EAR médio dos dois olhos cai abruptamente quando o olho se fecha. O limiar padrão é EAR < 0,22 "
        "por no mínimo 2 frames consecutivos. A duração da piscada é medida em ms: piscadas com duração "
        "< 150 ms são classificadas como involuntárias (reflexo palpebral), acima disso como voluntárias "
        "(Stern et al., 1984). A taxa é calculada por janela de 60 segundos."
    ))
    y -= 0.010

    y = subsection(fig, y, "5.4  Fadiga Ocular — FatigueDetector (PERCLOS + EAR Trend)")
    y = body(fig, y, (
        "O score de fadiga combina três componentes com pesos calibrados:\n"
        "\n"
        "  (1) PERCLOS (peso 0,50): fração de frames com eyeBlink ≥ 0,80 na janela de 60 s.\n"
        "  (2) EAR Trend (peso 0,35): coeficiente angular da regressão linear da EAR nos últimos 30 s; "
        "valor negativo indica fechamento progressivo (sonolência).\n"
        "  (3) eyeWide blendshape (peso 0,15): média dos blendshapes eyeWideLeft e eyeWideRight; "
        "valores altos indicam compensação consciente à sonolência (olhos arregalados).\n"
        "\n"
        "O score final [0, 1] é mapeado para quatro níveis: alert (< 0,20), mild (< 0,40), "
        "moderate (< 0,60) e severe (≥ 0,60)."
    ))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_metodologia_2(pdf, page_num):
    """Gaze + Pose + Emoções + Microexpressões + Respiração + Assimetria."""
    fig = new_page()
    page_header(fig, "5. Metodologia — Olhar · Pose · Emoções · Microexpressões · Respiração · Assimetria")
    page_footer(fig, page_num)

    y = 0.935

    y = subsection(fig, y, "5.5  Estimativa de Direção do Olhar — GazeEstimator")
    y = body(fig, y, (
        "A posição da íris é obtida pelos landmarks extras 468–477 do MediaPipe. A posição horizontal "
        "normalizada da íris dentro dos limites dos cantos do olho é calculada para ambos os olhos e "
        "combinada. Valores fora do intervalo [0,35; 0,65] horizontal e [0,35; 0,65] vertical indicam "
        "desvio. A métrica on_screen acumula o percentual de frames com olhar classificado como 'center'. "
        "Nota: esta é estimativa de direção, não de ponto exato na tela (gaze point-of-regard), que "
        "exigiria calibração com alvos visuais conhecidos."
    ))
    y -= 0.008

    y = subsection(fig, y, "5.6  Pose de Cabeça em 3D — HeadPoseEstimator (solvePnP)")
    y = body(fig, y, (
        "Seis landmarks estáveis (nariz, queixo, cantos dos olhos, cantos da boca) são combinados com "
        "um modelo 3D genérico de crânio. O cv2.solvePnP com método RANSAC resolve a "
        "correspondência 2D-3D e retorna o vetor de rotação, convertido para ângulos de Euler "
        "(yaw, pitch, roll) via cv2.Rodrigues. A matriz de câmera é aproximada por modelo pinhole "
        "com distância focal f = largura do frame — suficiente para webcam sem calibração formal."
    ))
    y -= 0.008

    y = subsection(fig, y, "5.7  Emoções — EmotionDetector (Blendshapes ARKit)")
    y = body(fig, y, (
        "Sete grupos de emoção são mapeados a partir dos 52 blendshapes ARKit sem necessidade de modelo "
        "adicional: alegria (mouthSmile, cheekSquint), surpresa (eyeWide, jawOpen), medo (browInnerUp, "
        "mouthStretch), nojo (noseSneer, mouthFrown), raiva (browDown, eyeSquint), tristeza "
        "(browInnerUp, mouthFrown, cheekPuff) e neutro (ausência de padrão dominante). A emoção "
        "dominante em cada frame é determinada pelo score máximo entre os grupos."
    ))
    y -= 0.008

    y = subsection(fig, y, "5.8  Microexpressões — MicroexpressionDetector")
    y = body(fig, y, (
        "Janela deslizante de 12 frames (~400 ms a 30 fps). Para cada grupo emocional, o baseline é a "
        "média dos primeiros 6 frames da janela e o pico é o valor máximo nos últimos 6. Uma "
        "microexpressão é detectada quando: (a) Δscore = pico − baseline ≥ 0,15, e (b) a duração "
        "do spike é ≤ 6 frames consecutivos. O critério de duração é fundamental para distinguir "
        "microexpressões de expressões plenas, conforme Ekman & Friesen (1969)."
    ))
    y -= 0.008

    y = subsection(fig, y, "5.9  Análise de Respiração — BreathingAnalyzer")
    y = body(fig, y, (
        "A respiração é estimada indiretamente pela variação temporal da largura nasal normalizada, "
        "calculada como a distância euclidiana entre os landmarks 64 (narina esquerda) e 294 "
        "(narina direita) em coordenadas normalizadas. A taxa respiratória é obtida por contagem de "
        "ciclos (cruzamentos do sinal pela média) em janela de 30 segundos. A profundidade é "
        "classificada em shallow / normal / deep com base na amplitude pico-a-vale."
    ))
    y -= 0.008

    y = subsection(fig, y, "5.10  Assimetria Facial — AsymmetryDetector")
    y = body(fig, y, (
        "Dez pares de blendshapes bilaterais são analisados: olhos (eyeBlink, eyeSquint, eyeWide), "
        "boca (mouthSmile, mouthFrown), sobrancelhas (browDown, browOuterUp), bochechas (cheekSquint, "
        "cheekPuff) e nariz (noseSneer). Para cada par:\n"
        "\n"
        "    delta = |score_esquerdo − score_direito|\n"
        "    score_normalizado = min(delta / 0,30, 1,0)\n"
        "\n"
        "O score global é a média dos 10 pares. O lado dominante (esquerdo/direito/simétrico) é "
        "determinado pela soma acumulada dos deltas. Score < 0,10 = simétrico; > 0,40 = assimetria "
        "pronunciada (potencialmente relevante em contexto clínico)."
    ))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resultados_1(pdf, page_num, df):
    """EAR + Gaze."""
    fig = new_page()
    page_header(fig, "6. Resultados — EAR e Piscadas · Distribuição do Olhar")
    page_footer(fig, page_num)

    fig.text(0.08, 0.930, "6. Resultados — Sessão de Exemplo",
             fontsize=12, fontweight="bold", color=PRIMARY, va="top")
    hline(fig, 0.912, color=ACCENT)

    # Filtrar pessoa 0
    p0 = df[df["person_id"] == 0].copy() if "person_id" in df.columns else df.copy()
    persons = df["person_id"].unique() if "person_id" in df.columns else [0]

    # ── Gráfico 1: EAR ao longo do tempo ─────────────────────────────────────
    ax1 = fig.add_axes([0.08, 0.665, 0.84, 0.215])
    for pid in sorted(persons)[:4]:
        sub = df[df["person_id"] == pid] if "person_id" in df.columns else df
        color = PERSON_COLORS[pid % len(PERSON_COLORS)]
        ax1.plot(sub["timestamp"], sub["ear"], lw=0.7, alpha=0.85,
                 color=color, label=f"Pessoa {pid}")
        blinks = sub[sub["blink_occurred"] == True]
        ax1.scatter(blinks["timestamp"], blinks["ear"],
                    color=color, s=18, zorder=5, marker="v")

    ax1.axhline(0.22, color=ACCENT, lw=1.0, ls="--", alpha=0.7, label="Limiar EAR (0,22)")
    ax1.set_xlabel("Tempo (s)", fontsize=8, color=GRAY)
    ax1.set_ylabel("EAR", fontsize=8, color=GRAY)
    ax1.set_title("Eye Aspect Ratio (EAR) ao longo do tempo  —  ▼ indica piscada detectada",
                  fontsize=8.5, color=PRIMARY, fontweight="bold", pad=5)
    ax1.tick_params(labelsize=7.5, colors=GRAY)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=7.5, framealpha=0.5)
    ax1.set_facecolor(LIGHT_BG)

    # ── Gráficos 2: Distribuição do olhar (pizza) ─────────────────────────────
    gaze_counts = p0["gaze_direction"].value_counts() if "gaze_direction" in p0.columns else pd.Series()
    if not gaze_counts.empty:
        ax2 = fig.add_axes([0.08, 0.390, 0.38, 0.235])
        colors_pie = [BLUE, GREEN, ORANGE, PURPLE, TEAL, GRAY]
        wedges, texts, autotexts = ax2.pie(
            gaze_counts.values,
            labels=gaze_counts.index,
            autopct="%1.1f%%",
            colors=colors_pie[:len(gaze_counts)],
            startangle=90,
            textprops={"fontsize": 7.5},
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax2.set_title("Distribuição do Olhar\n(Pessoa 0)", fontsize=8.5,
                      color=PRIMARY, fontweight="bold")

    # ── Gráfico 3: on_screen ao longo do tempo ────────────────────────────────
    if "on_screen" in p0.columns:
        ax3 = fig.add_axes([0.54, 0.390, 0.38, 0.235])
        on_pct = p0.groupby(p0["timestamp"] // 5)["on_screen"].mean() * 100
        bars = ax3.bar(on_pct.index * 5, on_pct.values, width=4,
                       color=[GREEN if v >= 70 else ORANGE for v in on_pct.values],
                       alpha=0.85)
        ax3.axhline(70, color=ACCENT, lw=1.0, ls="--", alpha=0.7, label="Meta 70%")
        ax3.set_xlabel("Tempo (s)", fontsize=8, color=GRAY)
        ax3.set_ylabel("% na tela", fontsize=8, color=GRAY)
        ax3.set_ylim(0, 105)
        ax3.set_title("Atenção à Tela por janela de 5 s\n(Pessoa 0)", fontsize=8.5,
                      color=PRIMARY, fontweight="bold")
        ax3.tick_params(labelsize=7.5, colors=GRAY)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.legend(fontsize=7.5)
        ax3.set_facecolor(LIGHT_BG)

    # ── Análise textual ──────────────────────────────────────────────────────
    y = 0.360
    blink_counts = {}
    blink_rates = {}
    for pid in sorted(persons):
        sub = df[df["person_id"] == pid] if "person_id" in df.columns else df
        blinks = sub[sub["blink_occurred"] == True]
        duration = sub["timestamp"].max() - sub["timestamp"].min()
        blink_counts[pid] = len(blinks)
        blink_rates[pid] = len(blinks) / (duration / 60) if duration > 0 else 0

    p0_onscreen = p0["on_screen"].mean() * 100 if "on_screen" in p0.columns else 0
    p0_gaze_dom = gaze_counts.index[0] if not gaze_counts.empty else "N/A"

    y = body(fig, y, (
        f"Análise (Pessoa 0): {blink_counts.get(0, 0)} piscadas detectadas "
        f"(taxa: {blink_rates.get(0, 0):.1f}/min). Direção predominante do olhar: {p0_gaze_dom}. "
        f"Percentual de atenção à tela: {p0_onscreen:.1f}%. "
        "O gráfico de barras mostra a atenção por janelas de 5 s — barras verdes indicam ≥ 70% de atenção."
    ), size=8.5, color=GRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resultados_2(pdf, page_num, df):
    """Pose + Emoções."""
    fig = new_page()
    page_header(fig, "6. Resultados — Pose de Cabeça · Distribuição de Emoções")
    page_footer(fig, page_num)
    page_header(fig, "6. Resultados (cont.)")

    p0 = df[df["person_id"] == 0].copy() if "person_id" in df.columns else df.copy()

    # ── Pose ao longo do tempo ────────────────────────────────────────────────
    ax1 = fig.add_axes([0.08, 0.670, 0.84, 0.220])
    if all(c in p0.columns for c in ["head_yaw", "head_pitch", "head_roll"]):
        ax1.plot(p0["timestamp"], p0["head_yaw"], color=BLUE, lw=0.8, label="Yaw (giro horiz.)")
        # Normalize pitch for display
        pitch_disp = p0["head_pitch"] - p0["head_pitch"].median()
        ax1.plot(p0["timestamp"], pitch_disp, color=GREEN, lw=0.8, label="Pitch (inclinação vert., centrado)")
        ax1.plot(p0["timestamp"], p0["head_roll"], color=ORANGE, lw=0.8, label="Roll (rotação lateral)")
        ax1.axhline(0, color=GRAY, lw=0.5, ls="--")
        ax1.set_xlabel("Tempo (s)", fontsize=8, color=GRAY)
        ax1.set_ylabel("Graus (°)", fontsize=8, color=GRAY)
        ax1.set_title("Pose de Cabeça — Yaw / Pitch / Roll ao longo do tempo  (Pessoa 0)",
                      fontsize=8.5, color=PRIMARY, fontweight="bold", pad=5)
        ax1.tick_params(labelsize=7.5, colors=GRAY)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.legend(fontsize=7.5, framealpha=0.5)
        ax1.set_facecolor(LIGHT_BG)

    # ── Emoções ───────────────────────────────────────────────────────────────
    emotions = ["neutral", "happy", "angry", "sad", "surprise", "fear", "disgust"]
    em_colors = [GRAY, GOLD, ACCENT, BLUE, PURPLE, TEAL, GREEN]

    if "dominant_emotion" in p0.columns:
        em_counts = p0["dominant_emotion"].value_counts()
        ax2 = fig.add_axes([0.08, 0.395, 0.38, 0.230])
        present = [e for e in emotions if e in em_counts.index]
        vals = [em_counts.get(e, 0) for e in present]
        bar_colors = [em_colors[emotions.index(e)] for e in present]
        bars = ax2.barh(present, vals, color=bar_colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            pct = val / len(p0) * 100
            ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%", va="center", fontsize=7.5, color=GRAY)
        ax2.set_xlabel("Frames", fontsize=8, color=GRAY)
        ax2.set_title("Distribuição de Emoções\n(Pessoa 0)", fontsize=8.5,
                      color=PRIMARY, fontweight="bold")
        ax2.tick_params(labelsize=7.5, colors=GRAY)
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.set_facecolor(LIGHT_BG)

    # Emoções ao longo do tempo (scatter)
    if "dominant_emotion" in p0.columns:
        ax3 = fig.add_axes([0.54, 0.395, 0.38, 0.230])
        em_map = {e: i for i, e in enumerate(emotions)}
        p0_copy = p0.copy()
        p0_copy["em_idx"] = p0_copy["dominant_emotion"].map(em_map)
        scatter_colors = [em_colors[int(i)] if not np.isnan(i) else GRAY
                          for i in p0_copy["em_idx"]]
        ax3.scatter(p0_copy["timestamp"], p0_copy["em_idx"],
                    c=scatter_colors, s=4, alpha=0.6)
        ax3.set_yticks(range(len(emotions)))
        ax3.set_yticklabels(emotions, fontsize=7)
        ax3.set_xlabel("Tempo (s)", fontsize=8, color=GRAY)
        ax3.set_title("Emoção dominante ao longo do tempo\n(Pessoa 0)",
                      fontsize=8.5, color=PRIMARY, fontweight="bold")
        ax3.tick_params(labelsize=7.5, colors=GRAY)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.set_facecolor(LIGHT_BG)

    # ── Sudden movements ─────────────────────────────────────────────────────
    if "sudden_movement" in p0.columns:
        sudden = p0[p0["sudden_movement"] == True]
        n_sudden = len(sudden)
        duration = p0["timestamp"].max() - p0["timestamp"].min()
        rate = n_sudden / (duration / 60) if duration > 0 else 0
        y = 0.365
        y = body(fig, y, (
            f"Movimentos bruscos de cabeça detectados: {n_sudden} "
            f"({rate:.1f}/min). Pitch médio: {p0['head_pitch'].median():.1f}°  "
            f"Yaw médio: {p0['head_yaw'].mean():.1f}°  "
            f"Roll médio: {p0['head_roll'].mean():.1f}°. "
            "Emoção predominante: " +
            (p0["dominant_emotion"].mode()[0] if len(p0) > 0 else "N/A") + "."
        ), size=8.5, color=GRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resultados_3(pdf, page_num, df):
    """Fadiga + Microexpressões + Assimetria."""
    fig = new_page()
    page_header(fig, "6. Resultados — Fadiga · Microexpressões · Assimetria Facial")
    page_footer(fig, page_num)
    page_header(fig, "6. Resultados (cont.)")

    p0 = df[df["person_id"] == 0].copy() if "person_id" in df.columns else df.copy()

    # ── Fadiga ────────────────────────────────────────────────────────────────
    ax1 = fig.add_axes([0.08, 0.685, 0.50, 0.200])
    if "fatigue_score" in p0.columns:
        ax1.fill_between(p0["timestamp"], p0["fatigue_score"], alpha=0.3, color=ACCENT)
        ax1.plot(p0["timestamp"], p0["fatigue_score"], color=ACCENT, lw=1.0, label="Score fadiga")
        if "perclos" in p0.columns:
            ax1.plot(p0["timestamp"], p0["perclos"], color=ORANGE, lw=0.8,
                     ls="--", label="PERCLOS")
        ax1.axhline(0.20, color=GREEN, lw=0.8, ls=":", alpha=0.7)
        ax1.axhline(0.40, color=GOLD, lw=0.8, ls=":", alpha=0.7)
        ax1.axhline(0.60, color=ACCENT, lw=0.8, ls=":", alpha=0.7)
        ax1.text(0.5, 0.22, "mild", fontsize=6.5, color=GOLD, transform=ax1.transAxes)
        ax1.text(0.5, 0.42, "moderate", fontsize=6.5, color=ORANGE, transform=ax1.transAxes)
        ax1.text(0.5, 0.62, "severe", fontsize=6.5, color=ACCENT, transform=ax1.transAxes)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Tempo (s)", fontsize=8, color=GRAY)
        ax1.set_ylabel("Score [0–1]", fontsize=8, color=GRAY)
        ax1.set_title("Fadiga Ocular — Score e PERCLOS\n(Pessoa 0)",
                      fontsize=8.5, color=PRIMARY, fontweight="bold")
        ax1.tick_params(labelsize=7.5, colors=GRAY)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.legend(fontsize=7.5)
        ax1.set_facecolor(LIGHT_BG)

    # Fatigue level pie
    if "fatigue_level" in p0.columns:
        ax1b = fig.add_axes([0.62, 0.685, 0.30, 0.200])
        lv_counts = p0["fatigue_level"].value_counts()
        lv_colors = {"alert": GREEN, "mild": GOLD, "moderate": ORANGE, "severe": ACCENT}
        lv_c = [lv_colors.get(k, GRAY) for k in lv_counts.index]
        ax1b.pie(lv_counts.values, labels=lv_counts.index,
                 autopct="%1.1f%%", colors=lv_c,
                 textprops={"fontsize": 7.5})
        ax1b.set_title("Nível de Fadiga\n(Pessoa 0)", fontsize=8.5,
                       color=PRIMARY, fontweight="bold")

    # ── Assimetria ────────────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.08, 0.430, 0.84, 0.200])
    if "asymmetry_score" in p0.columns:
        ax2.fill_between(p0["timestamp"], p0["asymmetry_score"], alpha=0.25, color=PURPLE)
        ax2.plot(p0["timestamp"], p0["asymmetry_score"], color=PURPLE, lw=0.8,
                 label="Score assimetria global")
        ax2.axhline(0.10, color=GREEN, lw=0.8, ls="--", alpha=0.6, label="Simétrico (< 0,10)")
        ax2.axhline(0.40, color=ACCENT, lw=0.8, ls="--", alpha=0.6,
                    label="Assimetria pronunciada (> 0,40)")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Tempo (s)", fontsize=8, color=GRAY)
        ax2.set_ylabel("Score [0–1]", fontsize=8, color=GRAY)
        ax2.set_title("Assimetria Facial ao longo do tempo  (Pessoa 0)",
                      fontsize=8.5, color=PRIMARY, fontweight="bold")
        ax2.tick_params(labelsize=7.5, colors=GRAY)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(fontsize=7.5, framealpha=0.5)
        ax2.set_facecolor(LIGHT_BG)

    # ── Microexpressões ───────────────────────────────────────────────────────
    y = 0.400
    if "microexp_detected" in p0.columns:
        micros = p0[p0["microexp_detected"] == True]
        n_micro = len(micros)
        if n_micro > 0:
            em_dist = micros["microexp_emotion"].value_counts()
            fig.text(0.08, y, f"Microexpressões detectadas (Pessoa 0): {n_micro}",
                     fontsize=9, color=PRIMARY, fontweight="bold", va="top")
            y -= 0.025
            ax3 = fig.add_axes([0.08, y - 0.140, 0.38, 0.130])
            ax3.bar(em_dist.index, em_dist.values,
                    color=[PERSON_COLORS[i % len(PERSON_COLORS)]
                           for i in range(len(em_dist))],
                    alpha=0.85)
            ax3.set_xlabel("Tipo de microexpressão", fontsize=7.5, color=GRAY)
            ax3.set_ylabel("Ocorrências", fontsize=7.5, color=GRAY)
            ax3.set_title("Distribuição por tipo", fontsize=8, color=PRIMARY, fontweight="bold")
            ax3.tick_params(labelsize=7, colors=GRAY, rotation=20)
            ax3.spines[["top", "right"]].set_visible(False)
            ax3.set_facecolor(LIGHT_BG)

            if "microexp_intensity" in micros.columns:
                ax4 = fig.add_axes([0.54, y - 0.140, 0.38, 0.130])
                ax4.hist(micros["microexp_intensity"].dropna(), bins=15,
                         color=GOLD, alpha=0.85, edgecolor="white")
                ax4.set_xlabel("Intensidade (Δscore)", fontsize=7.5, color=GRAY)
                ax4.set_ylabel("Frequência", fontsize=7.5, color=GRAY)
                ax4.set_title("Distribuição de intensidade", fontsize=8,
                              color=PRIMARY, fontweight="bold")
                ax4.tick_params(labelsize=7, colors=GRAY)
                ax4.spines[["top", "right"]].set_visible(False)
                ax4.set_facecolor(LIGHT_BG)
        else:
            y = body(fig, y, "Nenhuma microexpressão detectada nesta sessão.", color=GRAY)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resultados_comparativo(pdf, page_num, df):
    """Comparativo multi-pessoa."""
    persons = df["person_id"].unique() if "person_id" in df.columns else [0]
    if len(persons) < 2:
        return

    fig = new_page()
    page_header(fig, "6. Resultados — Comparativo Multi-Pessoa")
    page_footer(fig, page_num)
    page_header(fig, "6. Resultados — Comparativo Multi-Pessoa")

    y = section_title(fig, 0.928, "6.4  Comparativo entre Pessoas")

    metrics = []
    for pid in sorted(persons):
        sub = df[df["person_id"] == pid] if "person_id" in df.columns else df
        dur = sub["timestamp"].max() - sub["timestamp"].min()
        blinks = sub[sub["blink_occurred"] == True] if "blink_occurred" in sub.columns else sub.iloc[0:0]
        row = {
            "person_id": pid,
            "frames": len(sub),
            "blinks": len(blinks),
            "blink_rate": len(blinks) / (dur / 60) if dur > 0 else 0,
            "on_screen": sub["on_screen"].mean() * 100 if "on_screen" in sub.columns else 0,
            "dom_emotion": sub["dominant_emotion"].mode()[0] if "dominant_emotion" in sub.columns and len(sub) > 0 else "N/A",
            "fatigue_mean": sub["fatigue_score"].mean() if "fatigue_score" in sub.columns else 0,
            "asym_mean": sub["asymmetry_score"].mean() if "asymmetry_score" in sub.columns else 0,
            "micros": sub["microexp_detected"].sum() if "microexp_detected" in sub.columns else 0,
        }
        metrics.append(row)

    mdf = pd.DataFrame(metrics)

    # Radar / bar comparison
    compare_cols = ["blink_rate", "on_screen", "fatigue_mean", "asym_mean"]
    compare_labels = ["Taxa piscadas/min", "Atenção tela (%)", "Score fadiga", "Score assimetria"]
    n_persons = len(mdf)

    ax = fig.add_axes([0.08, 0.560, 0.84, 0.310])
    x = np.arange(len(compare_labels))
    bw = 0.8 / n_persons
    for i, row in mdf.iterrows():
        vals = [row["blink_rate"], row["on_screen"] / 100, row["fatigue_mean"], row["asym_mean"]]
        color = PERSON_COLORS[int(row["person_id"]) % len(PERSON_COLORS)]
        bars = ax.bar(x + i * bw - (n_persons - 1) * bw / 2, vals, bw * 0.85,
                      label=f"Pessoa {int(row['person_id'])}", color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(compare_labels, fontsize=8.5)
    ax.set_title("Métricas comparativas entre pessoas (valores normalizados / taxa raw)",
                 fontsize=9, color=PRIMARY, fontweight="bold")
    ax.tick_params(labelsize=8, colors=GRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.set_facecolor(LIGHT_BG)

    # Table
    y = 0.530
    col_labels = ["Pessoa", "Frames", "Piscadas", "Taxa/min", "Atenção (%)", "Emoção dom.", "Fadiga", "Assimetria", "Microexp."]
    col_widths = [0.065, 0.065, 0.075, 0.075, 0.085, 0.115, 0.075, 0.090, 0.085]
    xs = [0.08]
    for w in col_widths[:-1]:
        xs.append(xs[-1] + w)

    # Header
    header_y = y
    for xi, label in zip(xs, col_labels):
        fig.text(xi, header_y, label, fontsize=7.5, color="white",
                 fontweight="bold", va="top",
                 bbox=dict(facecolor=PRIMARY, edgecolor="none",
                           boxstyle="square,pad=0.15", alpha=0.9))

    row_y = header_y - 0.038
    for idx, row in mdf.iterrows():
        bg = LIGHT_BG if idx % 2 == 0 else "white"
        color = PERSON_COLORS[int(row["person_id"]) % len(PERSON_COLORS)]
        vals = [
            f"P{int(row['person_id'])}",
            f"{int(row['frames'])}",
            f"{int(row['blinks'])}",
            f"{row['blink_rate']:.1f}",
            f"{row['on_screen']:.1f}",
            str(row["dom_emotion"]),
            f"{row['fatigue_mean']:.3f}",
            f"{row['asym_mean']:.3f}",
            f"{int(row['micros'])}",
        ]
        for xi, val in zip(xs, vals):
            fig.text(xi, row_y, val, fontsize=7.5, color=color if xi == xs[0] else "#333",
                     fontweight="bold" if xi == xs[0] else "normal",
                     va="top")
        row_y -= 0.028

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_limitacoes(pdf, page_num):
    fig = new_page()
    page_header(fig, "7. Limitações")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "7. Limitações")

    limitations = [
        ("Rastreamento de identidade multi-pessoa",
         "O centroid tracker falha quando duas pessoas cruzam em posições muito próximas (distância < 0,20 "
         "em coordenadas normalizadas), podendo trocar IDs momentaneamente. Solução robusta exigiria "
         "Re-identification (Re-ID) visual baseado em descritores faciais ou o algoritmo húngaro com "
         "custo combinado (distância + similaridade de aparência)."),
        ("Estimativa de respiração",
         "A variação da largura nasal é um proxy indireto da respiração — ruído postural (movimento lateral "
         "da cabeça) pode gerar falsos ciclos. A taxa respiratória estimada é adequada para detecção de "
         "padrões grosseiros, mas não substitui oximetria ou pneumografia de contato."),
        ("Microexpressões — threshold fixo",
         "O limiar de Δscore ≥ 0,15 foi definido empiricamente. Em condições de iluminação instável ou com "
         "movimentos bruscos de cabeça, ocorrem falsos positivos. Um limiar adaptativo por pessoa (baseline "
         "calibrado nos primeiros 10 s de sessão) reduziria esta limitação."),
        ("Pose de cabeça sem calibração de câmera",
         "A matriz de câmera aproximada (modelo pinhole com f = largura do frame) introduz erro sistemático "
         "nos ângulos absolutos de Euler. Os valores são precisos o suficiente para detectar movimentos "
         "relativos e tendências, mas não para medição angular absoluta."),
        ("Gaze — estimativa de direção, não point-of-regard",
         "A posição da íris nos cantos do olho estima a direção do olhar (centro/esquerda/direita/cima/baixo), "
         "mas não o ponto exato na tela. Point-of-regard requer calibração com alvos visuais conhecidos "
         "(ex: 5–9 pontos de calibração), inviável com webcam sem cooperação do usuário."),
        ("Emoções via blendshapes — acurácia limitada em casos extremos",
         "O mapeamento de blendshapes ARKit para emoções é robusto para expressões nítidas, mas apresenta "
         "ambiguidade em emoções compostas (ex: alegria irônica, surpresa negativa). Abordagens baseadas "
         "em CNN treinadas em datasets anotados (AffectNet, RAF-DB) atingem maior acurácia em casos limítrofes."),
        ("Sem validação quantitativa formal",
         "O sistema foi desenvolvido e testado com vídeo de webcam ao vivo sem comparação com ground truth "
         "rotulado. Validação formal exigiria benchmark em datasets públicos como MAHNOB-HCI (fadiga), "
         "300-W (landmarks), ou AffectNet (emoções), com métricas de acurácia, precisão e recall."),
        ("Desempenho — latência com múltiplas pessoas",
         "O MediaPipe é executado em modo IMAGE (sem rastreamento temporal interno), o que significa que "
         "o tempo de inferência escala linearmente com o número de rostos detectados. Para mais de 3-4 "
         "pessoas simultâneas em hardware modesto, o FPS efetivo pode cair abaixo de 15 fps."),
    ]

    for i, (title, text) in enumerate(limitations):
        y = subsection(fig, y, f"7.{i+1}  {title}", size=9.5)
        y = body(fig, y, text)
        y -= 0.012

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_conclusoes(pdf, page_num):
    fig = new_page()
    page_header(fig, "8. Conclusões")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "8. Conclusões")
    y = body(fig, y, (
        "Este trabalho demonstrou que é possível construir um sistema completo e modular de análise de "
        "comportamento visual a partir de webcam convencional, inteiramente em CPU, sem hardware "
        "especializado, usando exclusivamente bibliotecas open-source de Python."
    ))
    y -= 0.010

    y = body(fig, y, (
        "A escolha do MediaPipe Face Landmarker como fundação se mostrou acertada: os 478 landmarks e "
        "52 blendshapes ARKit fornecem uma base rica o suficiente para implementar oito dimensões de "
        "análise comportamental distintas — piscadas, gaze, pose, emoção, respiração, fadiga, "
        "microexpressões e assimetria — sem modelos adicionais de deep learning."
    ))
    y -= 0.010

    y = body(fig, y, (
        "O suporte a múltiplas pessoas simultâneas, implementado via centroid tracker com instâncias "
        "de detectores isoladas por pessoa, representa uma extensão relevante em relação às "
        "implementações típicas de análise facial única, com custo de implementação baixo e sem "
        "impacto na modularidade do sistema."
    ))
    y -= 0.018

    y = section_title(fig, y, "Contribuições principais")
    contributions = [
        "Pipeline completo de análise comportamental visual modular com 10 módulos independentes.",
        "Implementação do padrão PERCLOS (NHTSA, 1998) combinado com EAR trend para detecção de fadiga, "
        "sem dependência de modelo de deep learning.",
        "Detecção de microexpressões por spike detection em janela deslizante de 12 frames, "
        "inspirada nos trabalhos de Ekman & Friesen.",
        "Quantificação de assimetria facial por 10 pares de blendshapes bilaterais normalizados.",
        "Suporte nativo a múltiplas pessoas simultâneas com identidade persistente por centróide.",
        "Geração automática de relatório PDF individual por pessoa com 8 páginas de gráficos.",
        "Arquitetura extensível: adição de novo detector requer apenas implementar a interface "
        "update() e registrá-lo no PersonManager.",
    ]
    y = bullet_list(fig, y, contributions)
    y -= 0.018

    y = section_title(fig, y, "Trabalhos futuros")
    future = [
        "Validação quantitativa em datasets públicos rotulados (MAHNOB-HCI, 300-W, AffectNet).",
        "Substituição do centroid tracker por Re-ID visual baseado em embeddings faciais "
        "(ex: ArcFace, FaceNet) para rastreamento robusto com cruzamentos.",
        "Threshold adaptativo por pessoa para detecção de microexpressões (calibração nos primeiros 10 s).",
        "Estimativa de ponto de gaze (point-of-regard) com calibração assistida por 5 pontos na tela.",
        "Detecção de comportamentos anômalos em tempo real com alertas configuráveis "
        "(ex: PERCLOS > 0,35 por mais de 60 s dispara alerta de fadiga severa).",
        "Interface web (Flask / FastAPI + WebSocket) para visualização remota do HUD.",
    ]
    y = bullet_list(fig, y, future)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_referencias(pdf, page_num):
    fig = new_page()
    page_header(fig, "9. Referências Bibliográficas")
    page_footer(fig, page_num)

    y = section_title(fig, 0.928, "9. Referências Bibliográficas")

    refs = [
        ("[1]  Soukupová, T., & Čech, J. (2016).",
         "Real-Time Eye Blink Detection Using Facial Landmarks. "
         "21st Computer Vision Winter Workshop (CVWW), Rimske Toplice, Eslovênia. "
         "— Base teórica do Eye Aspect Ratio (EAR) para detecção de piscadas."),
        ("[2]  NHTSA — National Highway Traffic Safety Administration (1998).",
         "Drowsy Driver Detection and Warning System for Commercial Vehicle Drivers: "
         "Field Proportional Test Design, Analysis, and Progress. Technical Report DOT HS 808 964. "
         "— Definição e padronização do PERCLOS como métrica de sonolência ao volante."),
        ("[3]  Stern, J. A., Boyer, D., & Schroeder, D. (1994).",
         "Blink Rate: A Possible Measure of Fatigue. Human Factors, 36(2), 285–297. "
         "— Classificação de piscadas voluntárias/involuntárias por duração e taxa."),
        ("[4]  Ekman, P., & Friesen, W. V. (1969).",
         "Nonverbal Leakage and Clues to Deception. Psychiatry, 32(1), 88–106. "
         "— Descrição original das microexpressões faciais e sua duração característica (40–500 ms)."),
        ("[5]  Ekman, P., & Friesen, W. V. (1978).",
         "Facial Action Coding System: A Technique for the Measurement of Facial Movement. "
         "Consulting Psychologists Press, Palo Alto, CA. "
         "— Sistema de unidades de ação facial (AUs), base conceitual para mapeamento de blendshapes."),
        ("[6]  Ekman, P. (1992).",
         "An Argument for Basic Emotions. Cognition & Emotion, 6(3–4), 169–200. "
         "— Fundamentação das sete emoções básicas universais utilizadas no EmotionDetector."),
        ("[7]  Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019).",
         "MediaPipe: A Framework for Building Perception Pipelines. "
         "arXiv:1906.08172. Google LLC. "
         "— Framework base para detecção facial, landmarks e blendshapes."),
        ("[8]  Grishchenko, I., Bazarevsky, V., Zanfir, M., Zanfir, A., Gorban, A., Turner, J., ... & Grundmann, M. (2022).",
         "Attention Mesh: High-Fidelity Face Mesh Prediction in Real-Time. "
         "CVPR Workshops on Computer Vision for Augmented and Virtual Reality. "
         "— Arquitetura do modelo face_landmarker.task com 478 landmarks e blendshapes ARKit."),
        ("[9]  Gee, A., & Cipolla, R. (1994).",
         "Determining the Gaze of Faces in Images. Image and Vision Computing, 12(10), 639–647. "
         "— Fundamentos de estimativa de gaze e pose de cabeça em câmeras não calibradas."),
        ("[10]  Li, X., Pfister, T., Huang, X., Zhao, G., & Pietikäinen, M. (2013).",
         "A Spontaneous Micro-expression Database: Inducement, Collection and Baseline. "
         "IEEE FG 2013, 10th IEEE Int. Conf. on Automatic Face and Gesture Recognition. "
         "— Banco de dados de microexpressões espontâneas e critérios de detecção automática."),
        ("[11]  Apple Inc. (2017).",
         "ARKit Face Tracking — Blend Shapes Reference. Apple Developer Documentation. "
         "— Especificação dos 52 coeficientes ARKit utilizados nos detectores de emoção, "
         "fadiga, microexpressões e assimetria."),
        ("[12]  Rosebrock, A. (2018).",
         "Simple Object Tracking with OpenCV. PyImageSearch Blog. "
         "— Implementação de referência do centroid tracker para rastreamento multi-objeto."),
        ("[13]  OpenCV Team (2023).",
         "OpenCV 4.x Documentation — Camera Calibration and 3D Reconstruction (solvePnP). "
         "docs.opencv.org. "
         "— Algoritmo PnP com RANSAC e conversão de vetor de rotação (Rodrigues) para ângulos de Euler."),
        ("[14]  Hunter, J. D. (2007).",
         "Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95. "
         "— Biblioteca de visualização usada no HUD e na geração de relatórios PDF."),
        ("[15]  Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).",
         "Array Programming with NumPy. Nature, 585, 357–362. "
         "— Biblioteca de álgebra linear e operações vetoriais utilizada em todos os módulos."),
    ]

    for ref_id, text in refs:
        fig.text(0.08, y, ref_id, fontsize=8.2, color=PRIMARY, va="top",
                 fontweight="bold")
        y -= 0.018
        y = body(fig, y, text, size=8.2, color="#444444", x=0.12)
        y -= 0.008

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gera o relatório técnico do projeto.")
    parser.add_argument("--csv", default=None, help="Caminho para o CSV de métricas.")
    parser.add_argument("--output", default=None, help="Diretório de saída.")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df = load_csv(args.csv)
    if df is None:
        print("AVISO: Nenhum CSV de métricas encontrado. Gráficos de resultados serão omitidos.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"relatorio_tecnico_{ts}.pdf")

    total_pages = 16
    page_num = [0]

    def next_page():
        page_num[0] += 1
        return page_num[0]

    print(f"Gerando relatório técnico → {out_path}")

    with PdfPages(out_path) as pdf:
        d = pdf.infodict()
        d["Title"] = "Relatório Técnico — Visual Behavior Analysis Multi-Pessoa"
        d["Author"] = "Visual Behavior Analysis Pipeline"
        d["Subject"] = (
            "Análise de comportamento visual: piscadas, gaze, pose, emoções, "
            "fadiga, microexpressões, assimetria facial — suporte multi-pessoa"
        )
        d["Keywords"] = (
            "MediaPipe, EAR, PERCLOS, gaze estimation, facial asymmetry, "
            "microexpressions, multi-person tracking, OpenCV, Python"
        )

        page_cover(pdf)
        print("  [1/16] Capa")

        page_resumo(pdf, next_page())
        print("  [2/16] Resumo")

        page_introducao(pdf, next_page())
        print("  [3/16] Introdução e Objetivos")

        page_fundamentacao(pdf, next_page())
        print("  [4/16] Fundamentação Teórica")

        page_arquitetura(pdf, next_page())
        print("  [5/16] Arquitetura")

        page_metodologia_1(pdf, next_page())
        print("  [6/16] Metodologia — Detecção, Piscadas, Fadiga")

        page_metodologia_2(pdf, next_page())
        print("  [7/16] Metodologia — Gaze, Pose, Emoções, Micro, Respiração, Assimetria")

        if df is not None:
            page_resultados_1(pdf, next_page(), df)
            print("  [8/16] Resultados — EAR e Gaze")

            page_resultados_2(pdf, next_page(), df)
            print("  [9/16] Resultados — Pose e Emoções")

            page_resultados_3(pdf, next_page(), df)
            print("  [10/16] Resultados — Fadiga, Microexpressões, Assimetria")

            persons = df["person_id"].unique() if "person_id" in df.columns else [0]
            if len(persons) > 1:
                page_resultados_comparativo(pdf, next_page(), df)
                print("  [11/16] Resultados — Comparativo multi-pessoa")

        page_limitacoes(pdf, next_page())
        print(f"  [{page_num[0]+1}/16] Limitações")

        page_conclusoes(pdf, next_page())
        print(f"  [{page_num[0]}/16] Conclusões")

        page_referencias(pdf, next_page())
        print(f"  [{page_num[0]}/16] Referências")

    print(f"\nRelatório gerado com sucesso: {out_path}")
    print(f"Total de páginas: {page_num[0] + 1}")


if __name__ == "__main__":
    main()
