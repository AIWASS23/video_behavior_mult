"""
Visualização em tempo real e geração de relatório ao final da sessão.

Este módulo tem duas responsabilidades principais:

1. ``draw_overlay()`` — anota cada frame com um painel HUD semi-transparente
   exibindo as métricas atuais: piscadas, EAR, direção do olhar, pose da
   cabeça, emoção e um indicador visual de gaze.

2. ``generate_report()`` — chamado ao final da sessão; produz uma figura
   Matplotlib multi-painel (PNG) com oito gráficos sumarizando todos os
   atributos extraídos ao longo do tempo.
"""


import json
import os
from datetime import datetime
from typing import Optional

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .asymmetry_detector import AsymmetryResult
from .blink_detector import BlinkDetector
from .breathing_detector import BreathingResult
from .emotion_detector import EmotionResult
from .face_tracker import FaceData
from .fatigue_detector import FatigueResult
from .gaze_estimator import GazeResult
from .head_pose_estimator import HeadPose
from .metrics_collector import MetricsCollector
from .microexpression_detector import MicroexpressionResult


# Paleta de cores BGR para desenhos OpenCV
_C = {
    "white":  (255, 255, 255),
    "black":  (  0,   0,   0),
    "green":  (  0, 200,   0),
    "red":    (  0,   0, 220),
    "yellow": (  0, 200, 200),
    "orange": (  0, 140, 255),
    "blue":   (200, 100,   0),
    "purple": (180,   0, 180),
    "gray":   (100, 100, 100),
}

# Paleta de cores BGR por person_id (ciclica para N pessoas)
_PERSON_COLORS_BGR = [
    (  0, 200,   0),   # 0 verde
    (200, 200,   0),   # 1 ciano
    (  0, 200, 200),   # 2 amarelo
    (200,   0, 200),   # 3 magenta
    (255, 128,   0),   # 4 azul claro
    (  0, 128, 255),   # 5 laranja
    (128,   0, 255),   # 6 rosa
    (  0, 255, 128),   # 7 verde-limão
]

# Paleta de cores HEX para Matplotlib (mesma ordem)
_PERSON_COLORS_HEX = [
    "#00c800", "#c8c800", "#c8c800", "#c800c8",
    "#0080ff", "#ff8000", "#8000ff", "#00ff80",
]

def _person_color_bgr(person_id: int) -> tuple:
    """Retorna a cor BGR associada a um person_id (ciclica)."""
    return _PERSON_COLORS_BGR[person_id % len(_PERSON_COLORS_BGR)]

def _person_color_hex(person_id: int) -> str:
    """Retorna a cor HEX associada a um person_id (ciclica)."""
    return _PERSON_COLORS_HEX[person_id % len(_PERSON_COLORS_HEX)]


def _put(frame: np.ndarray, text: str, pos: tuple, color=_C["white"], scale: float = 0.5):
    """
    Desenha texto no frame com contorno preto para legibilidade universal.

    O contorno é desenhado primeiro com espessura 3 e cor preta; em seguida
    o texto principal é sobreposto com espessura 1 na cor desejada. Isso
    garante legibilidade sobre qualquer fundo, claro ou escuro.

    Parâmetros
    ----------
    frame : np.ndarray
        Frame BGR onde o texto será desenhado (modificado in-place).
    text : str
        Texto a ser exibido.
    pos : tuple
        Posição (x, y) em pixels do canto inferior esquerdo do texto.
    color : tuple, opcional
        Cor BGR do texto principal (padrão: branco).
    scale : float, opcional
        Escala da fonte FONT_HERSHEY_SIMPLEX (padrão: 0.5).
    """
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, _C["black"], 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,     1, cv2.LINE_AA)


_FATIGUE_COLORS = {
    "alert": _C["green"], "mild": _C["yellow"],
    "moderate": _C["orange"], "severe": _C["red"],
}


class Visualizer:
    """
    Gerenciador de visualização multi-pessoa: HUD ao vivo e relatório PDF final.

    Mantém estado mínimo por person_id para exibição temporária de
    microexpressões no HUD. Todos os outros dados são recebidos por parâmetro.
    """

    def __init__(self):
        """Inicializa o visualizador com dicionários de estado por pessoa."""
        # {person_id: (last_event, last_ts)}
        self._micro_state: dict[int, tuple] = {}

    def draw_overlay(
        self,
        frame: np.ndarray,
        persons: list[tuple[int, dict]],
        elapsed: float,
    ) -> np.ndarray:
        """
        Anota o frame com um painel HUD por pessoa detectada.

        Para cada pessoa rastreada, desenha um painel semi-transparente com
        suas métricas individuais, colorido com a cor associada ao seu ID.
        Um badge no canto superior direito exibe o total de pessoas na cena.

        Parâmetros
        ----------
        frame : np.ndarray
            Frame BGR original (modificado in-place).
        persons : list[tuple[int, dict]]
            Lista de ``(person_id, results_dict)`` produzida pelo loop principal.
            ``results_dict`` deve conter as chaves: ``face_data``,
            ``blink_detector``, ``gaze_result``, ``head_pose``,
            ``emotion_result``, ``breathing_result``, ``fatigue_result``,
            ``micro_result``, ``asym_result``.
        elapsed : float
            Tempo decorrido em segundos desde o início da sessão.

        Retorna
        -------
        np.ndarray
            Frame BGR com todas as anotações aplicadas.
        """
        h, w = frame.shape[:2]

        # Badge de contagem no canto superior direito
        n_persons = len(persons)
        badge_txt = f"Pessoas: {n_persons}"
        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        bx = w - tw - 16
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 6, 4), (w - 4, th + 14), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        _put(frame, badge_txt, (bx, th + 8), color=_C["white"], scale=0.6)

        # Tempo global
        _put(frame, f"t={elapsed:.1f}s", (10, 18), color=_C["gray"], scale=0.45)

        panel_w  = 310
        panel_h  = 290
        max_cols = max(1, w // (panel_w + 4))

        for slot, (person_id, r) in enumerate(persons):
            col = slot % max_cols
            row = slot // max_cols
            px  = col * (panel_w + 4)
            py  = row * (panel_h + 4) + 24  # abaixo do badge de tempo

            # Garante que o painel caiba verticalmente
            if py + panel_h > h:
                py = max(24, h - panel_h)

            p_color = _person_color_bgr(person_id)

            # Fundo semi-transparente do painel
            ov2 = frame.copy()
            cv2.rectangle(ov2, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
            cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
            # Borda colorida por pessoa
            cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), p_color, 2)

            face_data       = r.get("face_data")
            blink_detector  = r.get("blink_detector")
            gaze_result     = r.get("gaze_result")
            head_pose       = r.get("head_pose")
            emotion_result  = r.get("emotion_result")
            breathing_result= r.get("breathing_result")
            fatigue_result  = r.get("fatigue_result")
            micro_result    = r.get("micro_result")
            asym_result     = r.get("asym_result")

            # Landmarks do rosto com cor da pessoa
            if face_data and face_data.detected:
                self._draw_landmarks(frame, face_data, p_color)

            y = py + 18
            lx = px + 8

            _put(frame, f"Pessoa {person_id}", (lx, y), color=p_color, scale=0.55)
            y += 22

            # Piscadas
            if blink_detector:
                stats      = blink_detector.get_stats()
                blink_rate = blink_detector.get_blink_rate(elapsed)
                _put(frame, f"Pisc: {stats['total_blinks']}  ({blink_rate:.1f}/min)", (lx, y), scale=0.45)
                y += 18
                _put(frame,
                     f"  V:{stats['voluntary']} I:{stats['involuntary']} EAR:{blink_detector.current_ear:.3f}",
                     (lx, y), scale=0.40, color=_C["gray"])
                y += 20

            # Olhar
            if gaze_result:
                g_color = _C["green"] if gaze_result.on_screen else _C["yellow"]
                screen  = "TELA" if gaze_result.on_screen else "FORA"
                _put(frame, f"Olhar: {gaze_result.direction} [{screen}]", (lx, y), color=g_color, scale=0.44)
                y += 20

            # Pose
            if head_pose:
                pc = _C["green"] if head_pose.is_frontal else _C["orange"]
                _put(frame,
                     f"Y:{head_pose.yaw:+.0f} P:{head_pose.pitch:+.0f} R:{head_pose.roll:+.0f}",
                     (lx, y), color=pc, scale=0.42)
                y += 18
                if head_pose.sudden_movement:
                    _put(frame, "! MOV BRUSCO", (lx, y), color=_C["red"], scale=0.44)
                    y += 18

            # Emoção
            if emotion_result and emotion_result.dominant != "unknown":
                _put(frame, f"Emocao: {emotion_result.dominant}", (lx, y), color=_C["purple"], scale=0.44)
                y += 18

            # Respiração
            if breathing_result:
                b_color  = _C["green"] if breathing_result.is_reliable else _C["gray"]
                rate_str = (f"{breathing_result.respiratory_rate:.1f}/min"
                            if breathing_result.is_reliable else "...")
                _put(frame, f"Resp: {rate_str} [{breathing_result.breath_phase}]",
                     (lx, y), color=b_color, scale=0.42)
                y += 18

            # Fadiga
            if fatigue_result:
                f_color = _FATIGUE_COLORS.get(fatigue_result.fatigue_level, _C["gray"])
                _put(frame,
                     f"Fadiga: {fatigue_result.fatigue_level.upper()} ({fatigue_result.fatigue_score:.2f})",
                     (lx, y), color=f_color, scale=0.42)
                y += 18

            # Microexpressão (exibe por 2 s após detecção)
            if micro_result and micro_result.detected and micro_result.event:
                self._micro_state[person_id] = (micro_result.event, elapsed)
            state = self._micro_state.get(person_id)
            if state and (elapsed - state[1]) < 2.0:
                ev = state[0]
                _put(frame,
                     f"MicroExp: {ev.emotion} ({ev.intensity:.2f})",
                     (lx, y), color=_C["purple"], scale=0.42)
                y += 18

            # Assimetria
            if asym_result:
                a_color = _C["green"] if asym_result.asymmetry_score < 0.2 else _C["yellow"]
                _put(frame,
                     f"Assim: {asym_result.asymmetry_score:.2f} [{asym_result.dominant_side}]",
                     (lx, y), color=a_color, scale=0.42)
                y += 18

            # Indicador de olhar (canto inferior do painel)
            if gaze_result:
                ball_x = px + panel_w - 36
                ball_y = py + panel_h - 36
                self._draw_gaze_ball(frame, gaze_result, ball_x, ball_y, radius=22, color=p_color)

        return frame

    def _draw_landmarks(
        self, frame: np.ndarray, face_data: FaceData, color: tuple = _C["green"]
    ):
        """
        Marca pontos-chave do rosto e os centros das íris no frame.

        Parâmetros
        ----------
        frame : np.ndarray
            Frame BGR modificado in-place.
        face_data : FaceData
            Dados faciais com landmarks detectados.
        color : tuple
            Cor BGR dos landmarks (padrão: verde).
        """
        for idx in [1, 33, 263, 61, 291, 199]:
            cv2.circle(frame, face_data.get_landmark_px(idx), 2, color, -1)
        for iris_idx in [468, 473]:
            try:
                cv2.circle(frame, face_data.get_landmark_px(iris_idx), 5, color, 1)
            except Exception:
                pass

    def _draw_gaze_ball(
        self, frame: np.ndarray, g: GazeResult, cx: int, cy: int,
        radius: int = 28, color: tuple = _C["green"]
    ):
        """
        Desenha um indicador circular analógico da posição atual do olhar.

        Um círculo cinza representa o campo visual; uma bola colorida dentro
        dele indica a posição estimada da íris. Verde = atenção à tela;
        Amarelo = olhar desviado. Linhas cruzadas marcam o centro de referência.

        Parâmetros
        ----------
        frame : np.ndarray
            Frame BGR modificado in-place.
        g : GazeResult
            Resultado de olhar com razões horizontal e vertical normalizadas.
        cx : int
            Coordenada x do centro do indicador em pixels.
        cy : int
            Coordenada y do centro do indicador em pixels.
        """
        r = 28
        cv2.circle(frame, (cx, cy), r, _C["gray"], 2)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), _C["gray"], 1)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), _C["gray"], 1)

        dx = int((g.horizontal_ratio - 0.5) * 2 * r * 0.6)
        dy = int((g.vertical_ratio   - 0.5) * 2 * r * 0.6)
        dot = (
            max(cx - r + 6, min(cx + r - 6, cx + dx)),
            max(cy - r + 6, min(cy + r - 6, cy + dy)),
        )
        color = _C["green"] if g.on_screen else _C["yellow"]
        cv2.circle(frame, dot, 7, color, -1)
        cv2.circle(frame, (cx, cy), 2, _C["gray"], -1)

    # ── helpers internos para o relatório PDF ────────────────────────────────

    @staticmethod
    def _page_footer(fig: plt.Figure, page_num: int, total_pages: int, ts: str):
        """
        Adiciona rodapé com número de página e timestamp em todas as páginas.

        Parâmetros
        ----------
        fig : plt.Figure
            Figura Matplotlib à qual o rodapé será adicionado.
        page_num : int
            Número da página atual (começa em 1).
        total_pages : int
            Total de páginas do relatório.
        ts : str
            String de timestamp formatada exibida no rodapé.
        """
        fig.text(
            0.5, 0.01,
            f"Análise de Comportamento Visual  ·  {ts}  ·  Página {page_num}/{total_pages}",
            ha="center", va="bottom", fontsize=8, color="#888888",
        )


    # ── helpers internos para gráficos por pessoa ────────────────────────────

    def _section_header(
        self,
        pdf: "PdfPages",
        person_id: int,
        ps: dict,
        page_num: int,
        total_pages: int,
        ts: str,
        PAGE_SIZE: tuple,
        first_frame: Optional[np.ndarray] = None,
    ):
        """
        Gera uma página de cabeçalho de seção para uma pessoa.

        Exibe o ID da pessoa, a cor associada, a foto do primeiro frame em
        que foi detectada e uma tabela com as principais métricas individuais.

        Parâmetros
        ----------
        pdf : PdfPages
            Objeto PdfPages aberto onde a página será salva.
        person_id : int
            Identificador numérico da pessoa.
        ps : dict
            Sumário individual retornado por ``get_summary_per_person()``.
        page_num : int
            Número desta página no PDF.
        total_pages : int
            Total de páginas do relatório.
        ts : str
            Timestamp formatado para o rodapé.
        PAGE_SIZE : tuple
            Tamanho da página em polegadas.
        first_frame : np.ndarray ou None
            Crop BGR do primeiro frame em que a pessoa foi detectada.
            Se None, a área de imagem é omitida e a tabela ocupa toda a página.
        """
        p_hex  = _person_color_hex(person_id)
        has_img = first_frame is not None and first_frame.size > 0

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("#f8f9fa")

        # ── Cabeçalho ─────────────────────────────────────────────────────
        fig.text(0.5, 0.945, f"Pessoa {person_id}",
                 ha="center", fontsize=26, fontweight="bold", color=p_hex)
        fig.text(0.5, 0.895, "Métricas individuais da sessão",
                 ha="center", fontsize=13, color="#555555")

        # Linha colorida
        bar_ax = fig.add_axes([0.06, 0.878, 0.88, 0.006])
        bar_ax.set_facecolor(p_hex)
        bar_ax.set_axis_off()

        # ── Foto do primeiro frame (esquerda) ──────────────────────────────
        if has_img:
            img_rgb = first_frame[:, :, ::-1]   # BGR → RGB para matplotlib
            ax_img  = fig.add_axes([0.06, 0.08, 0.28, 0.78])
            ax_img.imshow(img_rgb)
            ax_img.axis("off")
            # Borda colorida ao redor da foto
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(p_hex)
                spine.set_linewidth(3)
            ax_img.set_title("Primeiro frame detectado",
                             fontsize=9, color="#555555", pad=4)
            table_left  = 0.37
            table_width = 0.57
        else:
            table_left  = 0.08
            table_width = 0.84

        # ── Tabela de métricas (direita ou centralizada) ──────────────────
        fmt = {
            "total_frames":           ("Frames detectados",         "{}"),
            "face_stability_pct":     ("Estabilidade facial",        "{:.1f} %"),
            "total_blinks":           ("Total de piscadas",          "{}"),
            "blink_rate_per_min":     ("Taxa de piscadas",           "{:.1f} / min"),
            "on_screen_pct":          ("Atenção à tela",             "{:.1f} %"),
            "dominant_emotion":       ("Emoção dominante",           "{}"),
            "avg_head_yaw_deg":       ("Yaw médio (abs)",            "{:.1f}°"),
            "avg_respiratory_rate":   ("Taxa respiratória",          "{:.1f} / min"),
            "avg_fatigue_score":      ("Score de fadiga",            "{:.2f}"),
            "dominant_fatigue":       ("Nível de fadiga",            "{}"),
            "total_microexpressions": ("Microexpressões detectadas", "{}"),
            "avg_asymmetry_score":    ("Assimetria facial",          "{:.3f}"),
        }
        rows = []
        for key, (label, pattern) in fmt.items():
            val = ps.get(key, "N/A")
            try:
                rows.append([label, pattern.format(val)])
            except Exception:
                rows.append([label, str(val)])

        ax_tbl = fig.add_axes([table_left, 0.08, table_width, 0.78])
        ax_tbl.axis("off")
        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=["Métrica", "Valor"],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.60)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(p_hex)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f0f4ff")
            cell.set_edgecolor("#dddddd")

        self._page_footer(fig, page_num, total_pages, ts)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _person_charts(
        self,
        pdf: "PdfPages",
        person_df,
        ps: dict,
        person_id: int,
        ts: str,
        start_page: int,
        total_pages: int,
        PAGE_SIZE: tuple,
        EMO_COLORS: dict,
    ) -> int:
        """
        Gera os gráficos individuais de uma pessoa e os salva no PDF.

        Produz 7 páginas de gráficos filtrados exclusivamente pelos dados
        daquela pessoa:

        1. EAR ao longo do tempo + eventos de piscada
        2. Pizza de piscadas + distribuição do olhar (barras)
        3. Mapa de calor do olhar + pose da cabeça
        4. Distribuição de emoções (barras) + linha do tempo
        5. Respiração (sinal + taxa)
        6. Fadiga (score + PERCLOS)
        7. Microexpressões + assimetria

        Parâmetros
        ----------
        pdf : PdfPages
            Objeto PdfPages aberto.
        person_df : pd.DataFrame
            Subconjunto do DataFrame filtrado para este person_id.
        ps : dict
            Sumário individual desta pessoa.
        person_id : int
            ID da pessoa.
        ts : str
            Timestamp formatado.
        start_page : int
            Número da primeira página deste bloco.
        total_pages : int
            Total de páginas do relatório.
        PAGE_SIZE : tuple
            Tamanho da página.
        EMO_COLORS : dict
            Mapeamento emoção → cor HEX.

        Retorna
        -------
        int
            Número da próxima página após este bloco.
        """
        det   = person_df[person_df["face_detected"]]
        t0    = person_df["timestamp"].iloc[0] if not person_df.empty else 0.0
        p_hex = _person_color_hex(person_id)
        pg    = start_page

        def footer(fig, offset=0):
            self._page_footer(fig, pg + offset, total_pages, ts)

        # ── Gráfico 1: EAR + piscadas ────────────────────────────────────
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        if not det.empty:
            ax.plot(det["timestamp"] - t0, det["ear"],
                    color=p_hex, lw=1.0, alpha=0.85, label="EAR")
        blinks = person_df[person_df["blink_occurred"]]
        if not blinks.empty:
            vol   = blinks[blinks["blink_type"] == "voluntary"]
            invol = blinks[blinks["blink_type"] == "involuntary"]
            ax.scatter(vol["timestamp"]   - t0, [0.12] * len(vol),
                       color="orange", s=50, zorder=5, marker="^", label="Voluntária")
            ax.scatter(invol["timestamp"] - t0, [0.16] * len(invol),
                       color="red",    s=40, zorder=5, marker="v", label="Involuntária")
        ax.axhline(0.22, color="red", ls="--", alpha=0.5, lw=1.2, label="Limiar EAR (0.22)")
        ax.set_title(f"Pessoa {person_id}  —  EAR e Eventos de Piscada", fontsize=14, pad=12)
        ax.set_xlabel("Tempo (s)"); ax.set_ylabel("EAR")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        # ── Gráfico 2: Pizza piscadas + barras olhar ─────────────────────
        fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=PAGE_SIZE)
        vol_n   = ps.get("voluntary_blinks", 0)
        invol_n = ps.get("involuntary_blinks", 0)
        total_b = vol_n + invol_n
        if total_b > 0:
            ax_pie.pie(
                [vol_n, invol_n],
                labels=[f"Voluntárias ({vol_n})", f"Involuntárias ({invol_n})"],
                colors=["#f4a261", "#e76f51"], autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 11},
                wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
            )
        else:
            ax_pie.text(0.5, 0.5, "Sem piscadas", ha="center", va="center",
                        fontsize=13, transform=ax_pie.transAxes)
        ax_pie.set_title(
            f"Piscadas\nTotal: {total_b}  ·  {ps.get('blink_rate_per_min', 0):.1f}/min",
            fontsize=13)

        gd = ps.get("gaze_distribution", {})
        if gd:
            sorted_gd  = dict(sorted(gd.items(), key=lambda x: -x[1]))
            bar_colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_gd)))
            bars = ax_bar.bar(sorted_gd.keys(), sorted_gd.values(),
                              color=bar_colors, edgecolor="white")
            ax_bar.bar_label(bars, fontsize=10, padding=3)
            total_f = sum(gd.values())
            for bar, val in zip(bars, sorted_gd.values()):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                            f"{val/total_f*100:.0f}%", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            ax_bar.tick_params(axis="x", rotation=30, labelsize=10)
            ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.set_title("Direção do Olhar", fontsize=13)
        ax_bar.set_ylabel("Frames")
        fig.suptitle(f"Pessoa {person_id}", fontsize=15, fontweight="bold", color=p_hex, y=1.01)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig, 1); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        # ── Gráfico 3: Heatmap olhar + pose cabeça ───────────────────────
        fig, (ax_heat, ax_pose) = plt.subplots(1, 2, figsize=PAGE_SIZE)
        if not det.empty:
            h2d = ax_heat.hist2d(det["gaze_h"], det["gaze_v"],
                                 bins=30, cmap="hot", density=True)
            fig.colorbar(h2d[3], ax=ax_heat, fraction=0.03).set_label("Densidade", fontsize=9)
            ax_heat.axvline(0.5, color="cyan", lw=1.0, ls="--", alpha=0.7)
            ax_heat.axhline(0.5, color="deepskyblue", lw=1.0, ls="--", alpha=0.7)
            ax_heat.invert_yaxis()
            on_pct = ps.get("on_screen_pct", 0)
            ax_heat.set_title(f"Mapa de Calor do Olhar\nAtenção à tela: {on_pct:.1f}%", fontsize=12)
            ax_heat.set_xlabel("Horizontal (0=esq · 1=dir)")
            ax_heat.set_ylabel("Vertical (0=cima · 1=baixo)")

            tx = det["timestamp"] - t0
            ax_pose.plot(tx, det["head_yaw"],   color="#e63946", lw=1.0, label="Yaw")
            ax_pose.plot(tx, det["head_pitch"], color="#457b9d", lw=1.0, label="Pitch")
            ax_pose.plot(tx, det["head_roll"],  color="#2d6a4f", lw=0.9, label="Roll", alpha=0.8)
            sudden = det[det["sudden_movement"]]
            if not sudden.empty:
                ax_pose.scatter(sudden["timestamp"] - t0, sudden["head_yaw"],
                                color="red", s=40, zorder=5, marker="x", label="Mov. brusco")
            ax_pose.axhline(0, color="gray", ls="--", alpha=0.4, lw=0.8)
            ax_pose.axhline( 20, color="orange", ls=":", alpha=0.5, lw=0.8)
            ax_pose.axhline(-20, color="orange", ls=":", alpha=0.5, lw=0.8)
            ax_pose.legend(fontsize=9); ax_pose.grid(alpha=0.3)
        ax_pose.set_title("Pose da Cabeça", fontsize=12)
        ax_pose.set_xlabel("Tempo (s)"); ax_pose.set_ylabel("Graus")
        fig.suptitle(f"Pessoa {person_id}", fontsize=15, fontweight="bold", color=p_hex, y=1.01)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig, 2); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        # ── Gráfico 4: Emoções ───────────────────────────────────────────
        fig, (ax_ebar, ax_etl) = plt.subplots(1, 2, figsize=PAGE_SIZE)
        ed = {k: v for k, v in ps.get("emotion_distribution", {}).items() if k != "unknown"}
        if ed:
            sorted_ed  = dict(sorted(ed.items(), key=lambda x: -x[1]))
            bar_colors = [EMO_COLORS.get(k, "#bdc3c7") for k in sorted_ed]
            bars = ax_ebar.bar(sorted_ed.keys(), sorted_ed.values(),
                               color=bar_colors, edgecolor="white")
            ax_ebar.bar_label(bars, fontsize=10, padding=3)
            total_e = sum(ed.values())
            for bar, val in zip(bars, sorted_ed.values()):
                ax_ebar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                             f"{val/total_e*100:.0f}%", ha="center", va="center",
                             fontsize=9, color="white", fontweight="bold")
            ax_ebar.tick_params(axis="x", labelsize=11)
            ax_ebar.grid(axis="y", alpha=0.3)
        else:
            ax_ebar.text(0.5, 0.5, "Sem dados", ha="center", va="center",
                         fontsize=13, transform=ax_ebar.transAxes)
        ax_ebar.set_title("Distribuição de Emoções", fontsize=12)
        ax_ebar.set_ylabel("Frames")

        emo_data = det[det["dominant_emotion"] != "unknown"] if not det.empty else det
        if not emo_data.empty:
            emo_labels = sorted(emo_data["dominant_emotion"].unique())
            emo_idx    = {e: i for i, e in enumerate(emo_labels)}
            tx_e = emo_data["timestamp"] - t0
            yv_e = emo_data["dominant_emotion"].map(emo_idx)
            ax_etl.scatter(tx_e, yv_e,
                           c=[EMO_COLORS.get(e, "#bdc3c7") for e in emo_data["dominant_emotion"]],
                           s=8, alpha=0.6, linewidths=0)
            ax_etl.set_yticks(range(len(emo_labels)))
            ax_etl.set_yticklabels(emo_labels, fontsize=10)
            ax_etl.set_ylim(-0.5, len(emo_labels) - 0.5)
            ax_etl.grid(axis="x", alpha=0.3)
        ax_etl.set_title("Linha do Tempo — Emoções", fontsize=12)
        ax_etl.set_xlabel("Tempo (s)")
        fig.suptitle(f"Pessoa {person_id}", fontsize=15, fontweight="bold", color=p_hex, y=1.01)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig, 3); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        # ── Gráfico 5: Respiração ────────────────────────────────────────
        fig, (ax_sig, ax_rate) = plt.subplots(2, 1, figsize=PAGE_SIZE,
                                              gridspec_kw={"height_ratios": [2, 1]})
        breath_det = det[det["nostril_width_norm"] > 0] if not det.empty else det
        if not breath_det.empty:
            tx_b = breath_det["timestamp"] - t0
            ax_sig.plot(tx_b, breath_det["nostril_width_norm"],
                        color="#cccccc", lw=0.7, alpha=0.8, label="Sinal bruto")
            smoothed = breath_det["nostril_width_norm"].rolling(5, center=True, min_periods=1).mean()
            ax_sig.plot(tx_b, smoothed, color=p_hex, lw=1.5, label="Suavizado")
            phase_colors = {"inhale": "#d4f1f9", "exhale": "#ffe0b2", "hold": "#f5f5f5"}
            prev_t, prev_ph = float(tx_b.iloc[0]), breath_det["breath_phase"].iloc[0]
            for t_val, ph in zip(tx_b.iloc[1:], breath_det["breath_phase"].iloc[1:]):
                if ph != prev_ph:
                    ax_sig.axvspan(prev_t, float(t_val),
                                   color=phase_colors.get(prev_ph, "#ffffff"), alpha=0.25, lw=0)
                    prev_t, prev_ph = float(t_val), ph
            ax_sig.axvspan(prev_t, float(tx_b.iloc[-1]),
                           color=phase_colors.get(prev_ph, "#ffffff"), alpha=0.25, lw=0)
            from matplotlib.patches import Patch
            ax_sig.legend(handles=ax_sig.get_legend_handles_labels()[0] + [
                Patch(color="#d4f1f9", alpha=0.6, label="Inspiração"),
                Patch(color="#ffe0b2", alpha=0.6, label="Expiração"),
            ], labels=ax_sig.get_legend_handles_labels()[1] + ["Inspiração", "Expiração"],
               fontsize=9, loc="upper right")
        else:
            ax_sig.text(0.5, 0.5, "Sinal não disponível (< 10 s de dados)",
                        ha="center", va="center", fontsize=12, transform=ax_sig.transAxes)
        ax_sig.set_title(f"Pessoa {person_id}  —  Sinal Respiratório", fontsize=13, pad=10)
        ax_sig.set_ylabel("Largura nasal norm."); ax_sig.grid(alpha=0.3)

        rate_data = det[det["respiratory_rate"] > 0] if not det.empty else det
        if not rate_data.empty:
            tx_r = rate_data["timestamp"] - t0
            ax_rate.plot(tx_r, rate_data["respiratory_rate"], color=p_hex, lw=1.2, label="Taxa")
            ax_rate.axhspan(12, 20, color="#c8e6c9", alpha=0.3, label="Normal (12–20)")
            avg_rr = ps.get("avg_respiratory_rate", 0)
            ax_rate.axhline(avg_rr, color="green", ls="--", lw=0.9, alpha=0.7,
                            label=f"Média: {avg_rr:.1f}")
            ax_rate.set_ylim(0, max(40, rate_data["respiratory_rate"].max() * 1.1))
            ax_rate.legend(fontsize=9)
        else:
            ax_rate.text(0.5, 0.5, "Taxa não calculada", ha="center", va="center",
                         fontsize=12, transform=ax_rate.transAxes)
        ax_rate.set_title("Taxa Respiratória Estimada", fontsize=12, pad=8)
        ax_rate.set_xlabel("Tempo (s)"); ax_rate.set_ylabel("Ciclos/min"); ax_rate.grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig, 4); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        # ── Gráfico 6: Fadiga ────────────────────────────────────────────
        fig, (ax_fscore, ax_perclos) = plt.subplots(2, 1, figsize=PAGE_SIZE,
                                                    gridspec_kw={"height_ratios": [2, 1]})
        fat_det = det[det["fatigue_level"] != "unknown"] if not det.empty else det
        if not fat_det.empty:
            tx_f = fat_det["timestamp"] - t0
            ax_fscore.plot(tx_f, fat_det["fatigue_score"], color=p_hex, lw=1.2, label="Score")
            ax_fscore.axhspan(0.00, 0.25, color="#d4edda", alpha=0.3, label="Alerta")
            ax_fscore.axhspan(0.25, 0.50, color="#fff3cd", alpha=0.3, label="Leve")
            ax_fscore.axhspan(0.50, 0.75, color="#ffe5d0", alpha=0.3, label="Moderada")
            ax_fscore.axhspan(0.75, 1.00, color="#f8d7da", alpha=0.3, label="Severa")
            ax_fscore.set_ylim(0, 1)
            ax_fscore.legend(fontsize=9, loc="upper right"); ax_fscore.grid(alpha=0.3)
            ax_perclos.fill_between(tx_f, fat_det["perclos"] * 100,
                                    color=p_hex, alpha=0.6, label="PERCLOS (%)")
            ax_perclos.axhline(15, color="red", ls="--", lw=1.0, alpha=0.7, label="Limiar (15%)")
            ax_perclos.set_ylim(0, max(20, fat_det["perclos"].max() * 100 * 1.1))
            ax_perclos.legend(fontsize=9); ax_perclos.grid(alpha=0.3)
        else:
            for ax in (ax_fscore, ax_perclos):
                ax.text(0.5, 0.5, "Sem dados de fadiga", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes)
        ax_fscore.set_title(f"Pessoa {person_id}  —  Fadiga Ocular (Score e PERCLOS)",
                            fontsize=13, pad=10)
        ax_fscore.set_ylabel("Score [0–1]")
        ax_perclos.set_title("PERCLOS ao Longo do Tempo", fontsize=12, pad=8)
        ax_perclos.set_xlabel("Tempo (s)"); ax_perclos.set_ylabel("PERCLOS (%)")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig, 5); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        # ── Gráfico 7: Microexpressões + Assimetria ──────────────────────
        fig, (ax_micro, ax_asym) = plt.subplots(1, 2, figsize=PAGE_SIZE)
        micros_df = det[det["microexp_detected"]] if not det.empty else det
        if not micros_df.empty:
            emo_list = sorted(micros_df["microexp_emotion"].unique())
            emo_map  = {e: i for i, e in enumerate(emo_list)}
            tx_m = micros_df["timestamp"] - t0
            yv_m = micros_df["microexp_emotion"].map(emo_map)
            sc   = ax_micro.scatter(
                tx_m, yv_m,
                c=micros_df["microexp_intensity"], cmap="Reds",
                vmin=0.1, vmax=0.5,
                s=micros_df["microexp_intensity"] * 400 + 30, alpha=0.8)
            fig.colorbar(sc, ax=ax_micro, fraction=0.03).set_label("Intensidade", fontsize=9)
            ax_micro.set_yticks(range(len(emo_list)))
            ax_micro.set_yticklabels(emo_list, fontsize=10)
            ax_micro.set_ylim(-0.5, len(emo_list) - 0.5)
            ax_micro.grid(alpha=0.3)
        else:
            ax_micro.text(0.5, 0.5, "Nenhuma microexpressão\ndetectada",
                          ha="center", va="center", fontsize=12, transform=ax_micro.transAxes)
        total_me = ps.get("total_microexpressions", 0)
        ax_micro.set_title(f"Microexpressões  (total: {total_me})", fontsize=12)
        ax_micro.set_xlabel("Tempo (s)"); ax_micro.set_ylabel("Emoção")

        asym_det = det[det["asymmetry_score"] > 0] if not det.empty else det
        if not asym_det.empty:
            tx_a = asym_det["timestamp"] - t0
            ax_asym.plot(tx_a, asym_det["asymmetry_score"], color=p_hex, lw=1.0, alpha=0.8)
            smoothed_a = asym_det["asymmetry_score"].rolling(15, center=True, min_periods=1).mean()
            ax_asym.plot(tx_a, smoothed_a, color=p_hex, lw=2.0, label="Média móvel")
            ax_asym.axhline(0.2, color="orange", ls="--", lw=1.0, alpha=0.7, label="Limiar (0.2)")
            avg_asym = ps.get("avg_asymmetry_score", 0)
            ax_asym.axhline(avg_asym, color="gray", ls=":", lw=1.0,
                            label=f"Média: {avg_asym:.3f}")
            ax_asym.set_ylim(0, max(0.5, asym_det["asymmetry_score"].max() * 1.1))
            ax_asym.legend(fontsize=9); ax_asym.grid(alpha=0.3)
        else:
            ax_asym.text(0.5, 0.5, "Sem dados de assimetria", ha="center", va="center",
                         fontsize=12, transform=ax_asym.transAxes)
        ax_asym.set_title("Assimetria Facial", fontsize=12)
        ax_asym.set_xlabel("Tempo (s)"); ax_asym.set_ylabel("Score [0–1]")

        fig.suptitle(f"Pessoa {person_id}", fontsize=15, fontweight="bold", color=p_hex, y=1.01)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        footer(fig, 6); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); pg += 1

        return pg

    def generate_report(
        self,
        metrics: MetricsCollector,
        summary: dict,
        output_dir: str,
        elapsed: float,
        first_frames: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Gera o relatório PDF com gráficos individuais por pessoa rastreada.

        Estrutura do PDF:
        - Página 1: Capa global com total de pessoas e tabela de resumo
        - Página 2: Tabela comparativa entre pessoas (apenas se > 1 pessoa)
        - Para cada pessoa: 1 página de sumário + 7 páginas de gráficos

        Os gráficos de cada pessoa são gerados exclusivamente a partir dos
        dados daquela pessoa — nenhum dado de outra pessoa é misturado.

        Parâmetros
        ----------
        metrics : MetricsCollector
            Coletor com o histórico completo da sessão.
        summary : dict
            Sumário global retornado por ``get_summary()``.
        output_dir : str
            Diretório onde o PDF será salvo.
        elapsed : float
            Duração total da sessão em segundos.

        Retorna
        -------
        str ou None
            Caminho do arquivo PDF gerado, ou None se não houver dados.
        """
        df = metrics.to_dataframe()
        if df.empty:
            print("[Relatório] Sem dados — relatório não gerado.")
            return None

        ts      = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        per_person   = metrics.get_summary_per_person(elapsed)
        person_ids   = summary.get("person_ids", [])
        person_count = summary.get("person_count", 1)

        # Páginas: 1 capa + (1 comparação se >1 pessoa) + por pessoa: 1 sumário + 7 gráficos
        PAGES_PER_PERSON = 8   # 1 sumário + 7 gráficos
        TOTAL_PAGES = 1 + (1 if person_count > 1 else 0) + person_count * PAGES_PER_PERSON
        PAGE_SIZE   = (11.69, 8.27)

        EMO_COLORS = {
            "happy": "#f4d03f", "sad": "#5dade2", "angry": "#e74c3c",
            "fear": "#8e44ad", "disgust": "#1abc9c",
            "surprise": "#f39c12", "neutral": "#95a5a6",
        }

        pdf_path = os.path.join(output_dir, f"report_{ts_file}.pdf")

        with PdfPages(pdf_path) as pdf:

            # ── Página 1: Capa global ────────────────────────────────────
            fig = plt.figure(figsize=PAGE_SIZE)
            fig.patch.set_facecolor("#f8f9fa")
            fig.add_axes([0, 0.78, 1, 0.22]).set_axis_off()
            fig.text(0.5, 0.90, "Relatório de Análise de Comportamento Visual",
                     ha="center", fontsize=22, fontweight="bold", color="#1a1a2e")
            fig.text(0.5, 0.83,
                     f"Gerado em {ts}   ·   {person_count} pessoa(s) rastreada(s)",
                     ha="center", fontsize=11, color="#555555")

            fmt_global = {
                "duration_seconds":   ("Duração da sessão",   "{:.1f} s"),
                "total_frames":       ("Total de frames",      "{}"),
                "person_count":       ("Pessoas rastreadas",   "{}"),
                "face_stability_pct": ("Estabilidade facial",  "{:.1f} %"),
                "total_blinks":       ("Total de piscadas",    "{}"),
                "blink_rate_per_min": ("Taxa de piscadas",     "{:.1f} / min"),
                "on_screen_pct":      ("Atenção à tela",       "{:.1f} %"),
                "sudden_movements":   ("Movimentos bruscos",   "{}"),
                "dominant_emotion":   ("Emoção dominante",     "{}"),
                "avg_respiratory_rate": ("Taxa respiratória",  "{:.1f} / min"),
                "avg_fatigue_score":  ("Score de fadiga médio","{:.2f}"),
            }
            rows = []
            for key, (label, pattern) in fmt_global.items():
                val = summary.get(key, "N/A")
                try:
                    rows.append([label, pattern.format(val)])
                except Exception:
                    rows.append([label, str(val)])

            ax_tbl = fig.add_axes([0.08, 0.04, 0.84, 0.70])
            ax_tbl.axis("off")
            tbl = ax_tbl.table(cellText=rows, colLabels=["Métrica", "Valor"],
                               loc="center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.55)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#1a1a2e")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#eef2ff")
                cell.set_edgecolor("#cccccc")

            self._page_footer(fig, 1, TOTAL_PAGES, ts)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            current_page = 2

            # ── Página 2: Comparação entre pessoas (se >1) ───────────────
            if person_count > 1:
                fig, ax = plt.subplots(figsize=PAGE_SIZE)
                ax.axis("off")
                ax.set_title(f"Comparação entre {person_count} Pessoas",
                             fontsize=15, fontweight="bold", pad=18, color="#1a1a2e")

                compare_keys = [
                    ("total_frames",           "Frames detectados"),
                    ("face_stability_pct",     "Estabilidade (%)"),
                    ("total_blinks",           "Piscadas"),
                    ("blink_rate_per_min",     "Taxa pisc. (/min)"),
                    ("on_screen_pct",          "Atenção tela (%)"),
                    ("dominant_emotion",       "Emoção dominante"),
                    ("avg_head_yaw_deg",       "Yaw médio (°)"),
                    ("avg_respiratory_rate",   "Resp. (/min)"),
                    ("avg_fatigue_score",      "Score fadiga"),
                    ("dominant_fatigue",       "Nível fadiga"),
                    ("total_microexpressions", "Microexpressões"),
                    ("avg_asymmetry_score",    "Assimetria"),
                ]
                col_labels = ["Métrica"] + [f"Pessoa {pid}" for pid in person_ids]
                table_data = []
                for key, label in compare_keys:
                    row = [label]
                    for pid in person_ids:
                        val = per_person.get(pid, {}).get(key, "N/A")
                        row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
                    table_data.append(row)

                tbl2 = ax.table(cellText=table_data, colLabels=col_labels,
                                loc="center", cellLoc="center")
                tbl2.auto_set_font_size(False); tbl2.set_fontsize(10); tbl2.scale(1, 1.6)
                for (r, c), cell in tbl2.get_celld().items():
                    if r == 0:
                        if c == 0:
                            cell.set_facecolor("#1a1a2e")
                            cell.set_text_props(color="white", fontweight="bold")
                        else:
                            pid = person_ids[c - 1]
                            cell.set_facecolor(_person_color_hex(pid))
                            cell.set_text_props(color="white", fontweight="bold")
                    elif r % 2 == 0:
                        cell.set_facecolor("#f9f9f9")
                    cell.set_edgecolor("#cccccc")

                fig.tight_layout(rect=[0, 0.04, 1, 0.96])
                self._page_footer(fig, current_page, TOTAL_PAGES, ts)
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
                current_page += 1

            # ── Bloco por pessoa: sumário + 7 gráficos ───────────────────
            for pid in person_ids:
                ps       = per_person.get(pid, {})
                person_df = df[df["person_id"] == pid].copy()

                # Página de sumário individual (com foto do primeiro frame)
                first_frame = (first_frames or {}).get(pid)
                self._section_header(pdf, pid, ps, current_page, TOTAL_PAGES, ts, PAGE_SIZE,
                                     first_frame=first_frame)
                current_page += 1

                # 7 páginas de gráficos
                current_page = self._person_charts(
                    pdf, person_df, ps, pid, ts,
                    current_page, TOTAL_PAGES, PAGE_SIZE, EMO_COLORS,
                )

            # Metadados do PDF
            info = pdf.infodict()
            info["Title"]   = "Relatório de Análise de Comportamento Visual — Multi-Pessoa"
            info["Author"]  = "Visual Behavior Analysis Pipeline"
            info["Subject"] = "Métricas individuais por pessoa: piscada, olhar, pose, emoção, respiração, fadiga"

        print(f"[Relatório] PDF salvo → {pdf_path}")
        return pdf_path
