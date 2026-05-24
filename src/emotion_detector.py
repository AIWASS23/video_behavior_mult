"""
Estimativa de emoções faciais baseada nos blendshapes do MediaPipe.

O MediaPipe Face Landmarker (Tasks API) produz 52 blendshapes ARKit por rosto,
cada um com uma pontuação em [0, 1] indicando a intensidade de uma expressão
muscular específica. Este módulo mapeia combinações dessas pontuações para
seis emoções básicas + neutro, sem nenhuma dependência de TensorFlow.

Mapeamento blendshape → emoção
-------------------------------
- happy    : mouthSmileLeft/Right, cheekSquintLeft/Right
- surprise : eyeWideLeft/Right, jawOpen, browOuterUpLeft/Right
- angry    : browDownLeft/Right, eyeSquintLeft/Right, noseSneerLeft/Right
- sad      : mouthFrownLeft/Right, browInnerUp
- disgust  : noseSneerLeft/Right, mouthLeft/Right (assimetria)
- fear     : eyeWideLeft/Right, browOuterUpLeft/Right, browInnerUp
- neutral  : complemento — 1 - max(todas acima)

Referência
----------
Apple ARKit Face Tracking:
https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
"""


import logging
from dataclasses import dataclass, field
from typing import Optional

from .face_tracker import FaceData

log = logging.getLogger(__name__)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
"""Rótulos de emoção suportados pelo estimador."""


@dataclass
class EmotionResult:
    """
    Resultado da estimativa de emoções para um frame.

    Atributos
    ----------
    dominant : str
        Emoção com maior pontuação estimada. Um dos valores em
        ``EMOTION_LABELS``.
    scores : dict[str, float]
        Dicionário com pontuação em [0, 1] para cada emoção.
        As pontuações são normalizadas para somar 1.0.
    """

    dominant: str
    scores: dict[str, float] = field(default_factory=dict)


# ── Mapeamento blendshape → componentes de emoção ────────────────────────────
# Cada emoção é calculada como média ponderada de blendshapes relevantes.
# Pesos maiores para os blendshapes mais discriminativos.
_EMOTION_MAP: dict[str, list[tuple[str, float]]] = {
    "happy": [
        ("mouthSmileLeft",    1.5),
        ("mouthSmileRight",   1.5),
        ("cheekSquintLeft",   0.5),
        ("cheekSquintRight",  0.5),
    ],
    "surprise": [
        ("eyeWideLeft",        1.0),
        ("eyeWideRight",       1.0),
        ("jawOpen",            1.5),
        ("browOuterUpLeft",    0.5),
        ("browOuterUpRight",   0.5),
    ],
    "angry": [
        ("browDownLeft",       1.5),
        ("browDownRight",      1.5),
        ("eyeSquintLeft",      0.5),
        ("eyeSquintRight",     0.5),
        ("noseSneerLeft",      0.5),
        ("noseSneerRight",     0.5),
    ],
    "sad": [
        ("mouthFrownLeft",     1.5),
        ("mouthFrownRight",    1.5),
        ("browInnerUp",        1.0),
    ],
    "disgust": [
        ("noseSneerLeft",      1.5),
        ("noseSneerRight",     1.5),
        ("mouthLeft",          0.5),
        ("mouthRight",         0.5),
    ],
    "fear": [
        ("eyeWideLeft",        1.0),
        ("eyeWideRight",       1.0),
        ("browOuterUpLeft",    1.0),
        ("browOuterUpRight",   1.0),
        ("browInnerUp",        0.5),
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


def _blendshapes_to_dict(blendshapes: list) -> dict[str, float]:
    """
    Converte a lista de objetos Category do MediaPipe em dicionário.

    Parâmetros
    ----------
    blendshapes : list
        Lista de objetos Category com atributos ``.category_name`` e ``.score``.

    Retorna
    -------
    dict[str, float]
        Mapeamento nome_blendshape → pontuação (0.0 a 1.0).
    """
    return {bs.category_name: bs.score for bs in blendshapes}


def _estimate_emotions(bs_dict: dict[str, float]) -> EmotionResult:
    """
    Estima as pontuações de emoção a partir de um dicionário de blendshapes.

    Para cada emoção, calcula a média ponderada dos blendshapes relevantes
    (normalizados pelos pesos totais). Em seguida, calcula o neutro como
    1 - max(demais emoções) e normaliza todas as pontuações para somar 1.

    Parâmetros
    ----------
    bs_dict : dict[str, float]
        Dicionário blendshape_name → score em [0, 1].

    Retorna
    -------
    EmotionResult
        Emoção dominante e pontuações normalizadas de todas as emoções.
    """
    raw: dict[str, float] = {}

    for emotion, components in _EMOTION_MAP.items():
        total_weight = sum(w for _, w in components)
        weighted_sum = sum(bs_dict.get(name, 0.0) * w for name, w in components)
        raw[emotion] = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Neutro é inversamente proporcional à expressividade geral
    raw["neutral"] = max(0.0, 1.0 - max(raw.values()))

    # Normaliza para somar 1
    total = sum(raw.values()) or 1.0
    scores = {k: v / total for k, v in raw.items()}

    dominant = max(scores, key=scores.get)
    return EmotionResult(dominant=dominant, scores=scores)


class EmotionDetector:
    """
    Estimador de emoções baseado em blendshapes faciais do MediaPipe.

    Não requer TensorFlow nem modelos externos. As emoções são derivadas
    diretamente das pontuações dos 52 blendshapes ARKit produzidos pelo
    FaceLandmarker, usando um mapeamento ponderado por regras.

    A precisão é adequada para classificação de estado emocional dominante
    em condições de iluminação e ângulo frontais. Para casos de uso que
    exijam maior acurácia, substituir pelo backend DeepFace
    (requer tensorflow compatível com a versão do mediapipe instalada).

    Parâmetros
    ----------
    sample_interval : int
        Intervalo em frames entre estimativas consecutivas.
        Valor 1 (padrão) processa todos os frames, pois o custo computacional
        dos blendshapes é muito baixo (sem rede neural adicional).
    enabled : bool
        Habilita ou desabilita o módulo (padrão: True).
    """

    def __init__(self, sample_interval: int = 1, enabled: bool = True):
        """
        Inicializa o estimador de emoções baseado em blendshapes.

        Parâmetros
        ----------
        sample_interval : int, opcional
            Frames entre estimativas consecutivas (padrão: 1).
        enabled : bool, opcional
            Se False, o módulo retorna sempre None (padrão: True).
        """
        self.sample_interval = sample_interval
        self.enabled         = enabled
        self._last: Optional[EmotionResult] = None

    def update(
        self,
        frame_bgr,       # mantido por compatibilidade de interface, não utilizado
        face_data: FaceData,
        frame_count: int,
    ) -> Optional[EmotionResult]:
        """
        Estima a emoção dominante para o frame atual usando blendshapes.

        O parâmetro ``frame_bgr`` é mantido por compatibilidade com a
        interface anterior (que usava DeepFace/FER), mas não é utilizado.
        A estimativa é feita exclusivamente a partir dos blendshapes contidos
        em ``face_data``.

        Parâmetros
        ----------
        frame_bgr : np.ndarray
            Frame BGR (não utilizado nesta implementação).
        face_data : FaceData
            Dados faciais do frame, incluindo os blendshapes do MediaPipe.
        frame_count : int
            Índice sequencial do frame (controla o sample_interval).

        Retorna
        -------
        EmotionResult ou None
            Estimativa de emoção (nova ou cacheada), ou None se o módulo
            estiver desabilitado, sem rosto detectado ou sem blendshapes.
        """
        if not self.enabled or not face_data.detected:
            return self._last

        if frame_count % self.sample_interval != 0:
            return self._last

        if face_data.blendshapes is None:
            return self._last

        try:
            bs_dict = _blendshapes_to_dict(face_data.blendshapes)
            self._last = _estimate_emotions(bs_dict)
        except Exception as exc:
            log.debug("EmotionDetector: erro na estimativa — %s", exc)

        return self._last
