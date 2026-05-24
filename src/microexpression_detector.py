"""
Detector de microexpressões via variações rápidas nos blendshapes ARKit.

Microexpressões são expressões faciais involuntárias de curtíssima duração
(40–200 ms), teorizado por Paul Ekman (1969) como pistas de emoções
suprimidas ou espontâneas. A detecção aqui é baseada em:

1. **Baseline dinâmica**: média dos scores de emoção na primeira metade
   de uma janela deslizante (~400 ms).
2. **Pico**: máximo dos scores na segunda metade da mesma janela.
3. **Critérios**: delta ≥ 0.15 acima da baseline E duração ≤ 200 ms
   (≤ 6 frames a 30 fps).

Os blendshapes ARKit são agrupados por emoção seguindo o mapeamento
de Ekman para as seis emoções universais: alegria, surpresa, medo,
desgosto, raiva e tristeza.
"""


from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .face_tracker import FaceData


# ── Mapeamento blendshape → emoção ─────────────────────────────────────────────
_EMOTION_BLENDSHAPES: dict[str, list[str]] = {
    "happiness": ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"],
    "surprise":  ["jawOpen", "eyeWideLeft", "eyeWideRight", "browOuterUpLeft", "browOuterUpRight"],
    "fear":      ["browInnerUp", "eyeWideLeft", "eyeWideRight", "mouthStretchLeft", "mouthStretchRight"],
    "disgust":   ["noseSneerLeft", "noseSneerRight", "mouthFrownLeft", "mouthFrownRight"],
    "anger":     ["browDownLeft", "browDownRight", "eyeSquintLeft", "eyeSquintRight"],
    "sadness":   ["mouthFrownLeft", "mouthFrownRight", "browInnerUp", "cheekSquintLeft"],
    "contempt":  ["mouthSmileLeft", "mouthDimpleLeft"],  # expressão assimétrica
}

# ── Parâmetros de detecção ─────────────────────────────────────────────────────
_WINDOW_FRAMES   = 12    # ~400 ms a 30 fps — janela de observação
_SPIKE_THRESHOLD = 0.15  # delta mínimo acima do baseline para ser microexpressão
_MAX_PEAK_FRAMES = 6     # ≤ 200 ms a 30 fps — duração máxima do pico


@dataclass
class MicroexpressionEvent:
    """
    Descreve um evento de microexpressão detectado.

    Atributos
    ----------
    emotion : str
        Emoção identificada no pico (ex.: ``'happiness'``, ``'fear'``).
    intensity : float
        Delta entre pico e baseline — mede a intensidade da microexpressão.
    duration_ms : float
        Duração estimada em milissegundos, baseada no número de frames
        acima do limiar de pico.
    timestamp : float
        Timestamp em segundos do frame onde o pico foi detectado.
    """

    emotion: str
    intensity: float
    duration_ms: float
    timestamp: float


@dataclass
class MicroexpressionResult:
    """
    Resultado do detector de microexpressões para um frame.

    Atributos
    ----------
    detected : bool
        ``True`` se uma microexpressão foi detectada neste frame.
    event : MicroexpressionEvent ou None
        Detalhes do evento detectado; None se ``detected`` for False.
    current_scores : dict[str, float]
        Score instantâneo por emoção para o frame atual.
    """

    detected: bool
    event: Optional[MicroexpressionEvent]
    current_scores: dict[str, float]


class MicroexpressionDetector:
    """
    Detector de microexpressões por análise de picos em blendshapes ARKit.

    Mantém um histórico deslizante de scores por emoção e compara uma
    baseline recente contra o pico observado, sinalizando picos rápidos
    que atendem aos critérios de duração e intensidade.

    Parâmetros
    ----------
    fps : float
        Taxa de quadros da fonte de vídeo (padrão: 30.0). Usada para
        converter a duração máxima em número de frames.
    """

    def __init__(self, fps: float = 30.0):
        """
        Inicializa os buffers de histórico de scores.

        Parâmetros
        ----------
        fps : float
            Taxa de quadros estimada da fonte de vídeo.
        """
        self._fps      = fps
        self._frame_ms = 1000.0 / fps
        self._history: deque[dict[str, float]] = deque(maxlen=_WINDOW_FRAMES)

    def _blendshape(self, face_data: FaceData, name: str) -> float:
        """
        Retorna o score de um blendshape pelo nome, ou 0.0 se ausente.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais com lista de blendshapes ARKit.
        name : str
            Nome do blendshape (ex.: ``'mouthSmile_L'``).

        Retorna
        -------
        float
            Score em [0, 1].
        """
        if not face_data.blendshapes:
            return 0.0
        for bs in face_data.blendshapes:
            if bs.category_name == name:
                return float(bs.score)
        return 0.0

    def _compute_scores(self, face_data: FaceData) -> dict[str, float]:
        """
        Calcula a média dos blendshapes de cada emoção no frame atual.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais do frame.

        Retorna
        -------
        dict[str, float]
            Score médio por emoção, em [0, 1].
        """
        return {
            em: float(np.mean([self._blendshape(face_data, n) for n in bs_names]))
            for em, bs_names in _EMOTION_BLENDSHAPES.items()
        }

    def update(
        self, face_data: FaceData, timestamp: float
    ) -> MicroexpressionResult:
        """
        Atualiza o histórico e detecta microexpressão no frame atual.

        Parâmetros
        ----------
        face_data : FaceData
            Dados de detecção facial do frame atual.
        timestamp : float
            Tempo em segundos desde o início da sessão.

        Retorna
        -------
        MicroexpressionResult
            Resultado com flag de detecção, evento (se detectado) e
            scores instantâneos por emoção.
        """
        if not face_data.detected:
            return MicroexpressionResult(detected=False, event=None, current_scores={})

        current = self._compute_scores(face_data)
        self._history.append(current)

        if len(self._history) < _WINDOW_FRAMES:
            return MicroexpressionResult(detected=False, event=None, current_scores=current)

        half = _WINDOW_FRAMES // 2

        # Baseline: média da primeira metade da janela
        baseline = {
            em: float(np.mean([self._history[i][em] for i in range(half)]))
            for em in _EMOTION_BLENDSHAPES
        }
        # Pico: máximo da segunda metade da janela
        recent_max = {
            em: float(np.max([self._history[i][em] for i in range(half, _WINDOW_FRAMES)]))
            for em in _EMOTION_BLENDSHAPES
        }

        # Encontra a emoção com maior delta acima da baseline
        best_emotion = None
        best_delta   = 0.0
        for em in _EMOTION_BLENDSHAPES:
            delta = recent_max[em] - baseline[em]
            if delta > best_delta:
                best_delta   = delta
                best_emotion = em

        event    = None
        detected = False

        if best_emotion and best_delta >= _SPIKE_THRESHOLD:
            # Conta frames recentes que sustentam o pico acima do meio-limiar
            half_threshold = _SPIKE_THRESHOLD * 0.5
            above = sum(
                1 for i in range(half, _WINDOW_FRAMES)
                if self._history[i][best_emotion] - baseline[best_emotion] >= half_threshold
            )
            if above <= _MAX_PEAK_FRAMES:
                detected = True
                event    = MicroexpressionEvent(
                    emotion=best_emotion,
                    intensity=round(best_delta, 4),
                    duration_ms=round(above * self._frame_ms, 1),
                    timestamp=timestamp,
                )

        return MicroexpressionResult(
            detected=detected,
            event=event,
            current_scores=current,
        )
