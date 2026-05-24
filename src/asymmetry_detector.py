"""
Detector de assimetria facial via comparação de blendshapes bilaterais ARKit.

A assimetria facial naturalmente aumenta em expressões emocionais espontâneas
versus expressões posadas (Ekman, 1980). Medir a diferença entre os lados
esquerdo e direito pode indicar expressões genuínas, tensão facial unilateral
ou condições neuromusculares.

O detector calcula:
- **Par delta**: |score_esquerdo − score_direito| para cada par de blendshapes.
- **Score por região**: média dos deltas dentro de uma região anatômica
  (olhos, boca, sobrancelhas, bochechas, nariz).
- **Score global**: média ponderada de todos os pares, normalizada em [0, 1].
- **Lado dominante**: identificado pela soma acumulada dos scores em cada lado.
"""


from dataclasses import dataclass
from typing import Optional

import numpy as np

from .face_tracker import FaceData


# ── Pares bilaterais de blendshapes (nome_esq, nome_dir, região) ───────────────
_PAIRS: list[tuple[str, str, str]] = [
    ("eyeBlinkLeft",    "eyeBlinkRight",    "eyes"),
    ("eyeWideLeft",     "eyeWideRight",     "eyes"),
    ("eyeSquintLeft",   "eyeSquintRight",   "eyes"),
    ("mouthSmileLeft",  "mouthSmileRight",  "mouth"),
    ("mouthFrownLeft",  "mouthFrownRight",  "mouth"),
    ("mouthDimpleLeft", "mouthDimpleRight", "mouth"),
    ("cheekSquintLeft", "cheekSquintRight", "cheeks"),
    ("browDownLeft",    "browDownRight",    "brows"),
    ("browOuterUpLeft", "browOuterUpRight", "brows"),
    ("noseSneerLeft",   "noseSneerRight",   "nose"),
]

# Score global máximo de referência para normalização (heurístico)
_NORM_MAX = 0.30

# Limiar abaixo do qual o rosto é considerado simétrico
_SYMMETRY_THRESHOLD = 0.05


@dataclass
class AsymmetryResult:
    """
    Resultado do detector de assimetria facial para um frame.

    Atributos
    ----------
    asymmetry_score : float
        Índice de assimetria global normalizado em [0, 1].
        0 = perfeitamente simétrico; 1 = assimetria máxima observável.
    dominant_side : str
        Lado com maior ativação acumulada: ``'left'``, ``'right'`` ou
        ``'symmetric'``.
    region_scores : dict[str, float]
        Assimetria média por região anatômica (``eyes``, ``mouth``,
        ``brows``, ``cheeks``, ``nose``). Valores em [0, 1].
    pair_deltas : dict[str, float]
        Diferença absoluta |esq − dir| para cada par de blendshapes.
        Chave = nome sem o sufixo ``_L``/``_R``.
    """

    asymmetry_score: float
    dominant_side: str
    region_scores: dict[str, float]
    pair_deltas: dict[str, float]


class AsymmetryDetector:
    """
    Detector de assimetria facial frame a frame.

    Compara pares de blendshapes esquerdo/direito para estimar o grau de
    assimetria em cinco regiões anatômicas e produz um índice global
    normalizado.

    Não mantém estado entre frames — cada chamada a ``update()`` é
    independente. Pode ser instanciado sem parâmetros.
    """

    def _blendshape(self, face_data: FaceData, name: str) -> float:
        """
        Extrai o score de um blendshape pelo nome.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais com lista de blendshapes ARKit.
        name : str
            Nome do blendshape (ex.: ``'mouthSmile_L'``).

        Retorna
        -------
        float
            Score em [0, 1]; 0.0 se o blendshape não for encontrado.
        """
        if not face_data.blendshapes:
            return 0.0
        for bs in face_data.blendshapes:
            if bs.category_name == name:
                return float(bs.score)
        return 0.0

    def update(self, face_data: FaceData) -> Optional[AsymmetryResult]:
        """
        Calcula o índice de assimetria facial para o frame atual.

        Parâmetros
        ----------
        face_data : FaceData
            Dados de detecção facial do frame atual.

        Retorna
        -------
        AsymmetryResult ou None
            Resultado com score global, lado dominante e breakdown por região,
            ou None se nenhum rosto ou blendshapes foram detectados.
        """
        if not face_data.detected or not face_data.blendshapes:
            return None

        pair_deltas: dict[str, float]      = {}
        region_sums: dict[str, list[float]] = {}
        left_total  = 0.0
        right_total = 0.0

        for left_name, right_name, region in _PAIRS:
            l_val = self._blendshape(face_data, left_name)
            r_val = self._blendshape(face_data, right_name)
            delta = abs(l_val - r_val)

            # Chave sem sufixo Left/Right
            key = left_name[:-4]
            pair_deltas[key] = round(delta, 4)
            region_sums.setdefault(region, []).append(delta)
            left_total  += l_val
            right_total += r_val

        raw_score = float(np.mean(list(pair_deltas.values())))
        asymmetry_score = float(np.clip(raw_score / _NORM_MAX, 0.0, 1.0))

        region_scores = {
            reg: round(float(np.mean(vals)), 4)
            for reg, vals in region_sums.items()
        }

        if raw_score < _SYMMETRY_THRESHOLD:
            dominant_side = "symmetric"
        elif left_total > right_total:
            dominant_side = "left"
        else:
            dominant_side = "right"

        return AsymmetryResult(
            asymmetry_score=round(asymmetry_score, 4),
            dominant_side=dominant_side,
            region_scores=region_scores,
            pair_deltas=pair_deltas,
        )
