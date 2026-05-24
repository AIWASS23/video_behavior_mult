"""
Estimativa da direção do olhar a partir dos landmarks das íris do MediaPipe.

O MediaPipe disponibiliza 10 landmarks extras das íris (468–477) quando
refine_landmarks=True é usado no Face Mesh:
  Íris esquerda : 468 (centro), 469 (topo), 470 (direita), 471 (baixo), 472 (esquerda)
  Íris direita  : 473 (centro), 474 (topo), 475 (direita), 476 (baixo), 477 (esquerda)

Estratégia
----------
Para cada olho, a posição do centro da íris é normalizada dentro da caixa
delimitadora formada pelos cantos do olho, gerando uma razão em [0, 1] para
os eixos horizontal e vertical. Os dois olhos são combinados por média
simples para reduzir ruído de detecção.

Convenção de coordenadas (espaço de imagem)
-------------------------------------------
  horizontal : 0 = extremidade esquerda do olho, 1 = extremidade direita
  vertical   : 0 = extremidade superior,          1 = extremidade inferior

Nota sobre espelhamento
-----------------------
Câmeras de webcam geralmente espelham a imagem (modo selfie). Os rótulos
de direção seguem o espaço da imagem; em gravações não espelhadas, o eixo
horizontal pode precisar ser invertido. Consulte o README para detalhes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .face_tracker import FaceData


@dataclass
class GazeResult:
    """
    Resultado da estimativa de direção do olhar para um único frame.

    Atributos
    ----------
    direction : str
        Direção textual do olhar no espaço da imagem.
        Valores possíveis: ``'center'``, ``'left'``, ``'right'``, ``'up'``,
        ``'down'``, ``'up-left'``, ``'up-right'``, ``'down-left'``,
        ``'down-right'``, ``'unknown'``.
    horizontal_ratio : float
        Posição horizontal normalizada da íris em [0, 1].
        0 = íris na borda esquerda do olho; 1 = borda direita.
        Valor 0.5 indica olhar centrado horizontalmente.
    vertical_ratio : float
        Posição vertical normalizada da íris em [0, 1].
        0 = íris na borda superior do olho; 1 = borda inferior.
        Valor 0.5 indica olhar centrado verticalmente.
    on_screen : bool
        Heurística de atenção à tela: True quando o olhar está
        aproximadamente centralizado horizontalmente e não muito abaixo,
        sugerindo que o usuário olha para o monitor/câmera.
    """

    direction: str
    horizontal_ratio: float
    vertical_ratio: float
    on_screen: bool


# ── Índices de landmarks do MediaPipe Face Mesh ───────────────────────────────
_L_IRIS_CTR  = 468   # centro da íris esquerda
_R_IRIS_CTR  = 473   # centro da íris direita

# Cantos dos olhos (outer = longe do nariz, inner = próximo ao nariz)
_L_EYE_OUTER = 362
_L_EYE_INNER = 263
_R_EYE_OUTER =  33
_R_EYE_INNER = 133

# Bordas superior e inferior dos olhos (para razão vertical)
_L_EYE_TOP = 386;  _L_EYE_BOT = 374
_R_EYE_TOP = 159;  _R_EYE_BOT = 145
# ─────────────────────────────────────────────────────────────────────────────


def _lm(landmarks, idx: int) -> np.ndarray:
    """
    Extrai as coordenadas normalizadas de um landmark como array NumPy.

    Parâmetros
    ----------
    landmarks : NormalizedLandmarkList
        Lista de landmarks do MediaPipe.
    idx : int
        Índice do landmark desejado.

    Retorna
    -------
    np.ndarray
        Array de forma (2,) com [x, y] em coordenadas normalizadas [0, 1].
    """
    lm = landmarks[idx]
    return np.array([lm.x, lm.y])


def _ratio(iris_coord: float, edge_a: float, edge_b: float) -> float:
    """
    Calcula a razão normalizada de posição da íris entre dois cantos do olho.

    A ordem de ``edge_a`` e ``edge_b`` é arbitrária; a função identifica
    automaticamente o mínimo e o máximo para garantir um resultado em [0, 1].

    Parâmetros
    ----------
    iris_coord : float
        Coordenada da íris no eixo de interesse (x ou y), normalizada [0, 1].
    edge_a : float
        Coordenada de um dos cantos do olho no mesmo eixo.
    edge_b : float
        Coordenada do outro canto do olho no mesmo eixo.

    Retorna
    -------
    float
        Razão em [0, 1]. Retorna 0.5 se a distância entre as bordas for
        desprezível (evita divisão por zero em detecções degeneradas).
    """
    lo, hi = min(edge_a, edge_b), max(edge_a, edge_b)
    span = hi - lo
    if span < 1e-6:
        return 0.5
    return float(np.clip((iris_coord - lo) / span, 0.0, 1.0))


class GazeEstimator:
    """
    Estimador de direção do olhar baseado na posição relativa das íris.

    Combina as razões de posição horizontal e vertical de ambos os olhos
    e mapeia para um dos nove rótulos de direção. A estimativa é computada
    inteiramente a partir dos landmarks do MediaPipe, sem câmera calibrada
    nem modelo de regressão adicional.

    Parâmetros
    ----------
    h_threshold : float
        Limiar de desvio lateral. Razões abaixo de ``h_threshold`` indicam
        olhar para a esquerda (no espaço de imagem); acima de
        ``1 - h_threshold`` indicam direita (padrão: 0.35).
    v_threshold : float
        Limiar de desvio vertical. Razões abaixo de ``v_threshold`` indicam
        olhar para cima; acima de ``1 - v_threshold`` indicam para baixo
        (padrão: 0.35).
    """

    def __init__(
        self,
        h_threshold: float = 0.35,
        v_threshold: float = 0.35,
    ):
        """
        Inicializa o estimador com os limiares de classificação direcional.

        Parâmetros
        ----------
        h_threshold : float, opcional
            Limiar horizontal de desvio do olhar (padrão: 0.35).
        v_threshold : float, opcional
            Limiar vertical de desvio do olhar (padrão: 0.35).
        """
        self.h_threshold = h_threshold
        self.v_threshold = v_threshold
        self.current: GazeResult = GazeResult("center", 0.5, 0.5, True)

    def update(self, face_data: FaceData) -> GazeResult:
        """
        Estima a direção do olhar para o frame atual.

        Extrai as posições das íris esquerda e direita, normaliza cada uma
        dentro da caixa delimitadora do respectivo olho, calcula a média e
        classifica em um dos rótulos de direção.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais do frame atual com landmarks do MediaPipe.
            Requer refine_landmarks=True no FaceTracker para dispor dos
            landmarks das íris (468, 473).

        Retorna
        -------
        GazeResult
            Resultado com direção textual, razões normalizadas e flag
            de atenção à tela. Retorna ``direction='unknown'`` e
            ``on_screen=False`` quando nenhum rosto é detectado.
        """
        if not face_data.detected:
            return GazeResult("unknown", 0.5, 0.5, False)

        lm = face_data.landmarks

        # ── Olho esquerdo ─────────────────────────────────────────────────
        l_iris  = _lm(lm, _L_IRIS_CTR)
        l_outer = _lm(lm, _L_EYE_OUTER)
        l_inner = _lm(lm, _L_EYE_INNER)
        l_top   = _lm(lm, _L_EYE_TOP)
        l_bot   = _lm(lm, _L_EYE_BOT)

        l_h = _ratio(l_iris[0], l_outer[0], l_inner[0])
        l_v = _ratio(l_iris[1], l_top[1],   l_bot[1])

        # ── Olho direito ──────────────────────────────────────────────────
        r_iris  = _lm(lm, _R_IRIS_CTR)
        r_outer = _lm(lm, _R_EYE_OUTER)
        r_inner = _lm(lm, _R_EYE_INNER)
        r_top   = _lm(lm, _R_EYE_TOP)
        r_bot   = _lm(lm, _R_EYE_BOT)

        r_h = _ratio(r_iris[0], r_inner[0], r_outer[0])
        r_v = _ratio(r_iris[1], r_top[1],   r_bot[1])

        # ── Média dos dois olhos ──────────────────────────────────────────
        avg_h = (l_h + r_h) / 2.0
        avg_v = (l_v + r_v) / 2.0

        # ── Mapeamento para rótulos de direção ────────────────────────────
        if avg_h < self.h_threshold:
            h_dir = "left"
        elif avg_h > (1.0 - self.h_threshold):
            h_dir = "right"
        else:
            h_dir = "center"

        if avg_v < self.v_threshold:
            v_dir = "up"
        elif avg_v > (1.0 - self.v_threshold):
            v_dir = "down"
        else:
            v_dir = "center"

        if h_dir == "center" and v_dir == "center":
            direction = "center"
        elif h_dir == "center":
            direction = v_dir
        elif v_dir == "center":
            direction = h_dir
        else:
            direction = f"{v_dir}-{h_dir}"

        # Heurística: olhar centrado horizontalmente e não muito abaixo → "na tela"
        on_screen = h_dir == "center" and v_dir in ("center", "up")

        result = GazeResult(direction, avg_h, avg_v, on_screen)
        self.current = result
        return result
