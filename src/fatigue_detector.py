"""
Detector de fadiga ocular baseado em PERCLOS, EAR temporal e blendshapes.

Implementa o PERCLOS (Percentage of Eye Closure), métrica padronizada para
detecção de sonolência ao volante (NHTSA, 1998), adaptada para análise de
atenção em frente à tela. Combina três indicadores complementares:

- **PERCLOS**: proporção de frames em que os olhos estão ≥ 80% fechados.
- **Tendência do EAR**: slope linear do Eye Aspect Ratio na janela recente
  (slope negativo indica olhos fechando progressivamente).
- **eyeWide blendshape**: abertura relativa dos olhos segundo os modelos
  ARKit do MediaPipe (menor valor = menos abertos = mais fatigados).

Os três componentes são combinados em um ``fatigue_score`` normalizado [0, 1]
e mapeado em quatro níveis qualitativos: ``alert``, ``mild``, ``moderate``
e ``severe``.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .face_tracker import FaceData


# ── Parâmetros internos ────────────────────────────────────────────────────────
_CLOSURE_THRESHOLD = 0.80   # eyeBlink score ≥ este valor = olho fechado
_PERCLOS_WINDOW_S  = 60.0   # janela temporal do PERCLOS em segundos
_EAR_WINDOW_S      = 30.0   # janela para cálculo do slope do EAR


@dataclass
class FatigueResult:
    """
    Resultado do detector de fadiga para um único frame.

    Atributos
    ----------
    fatigue_level : str
        Classificação qualitativa: ``'alert'``, ``'mild'``, ``'moderate'``
        ou ``'severe'``.
    fatigue_score : float
        Pontuação contínua de fadiga no intervalo [0, 1]. 0 = totalmente
        alerta; 1 = fadiga severa.
    perclos : float
        Proporção de frames na janela de 60 s em que os olhos estavam
        ≥ 80% fechados. Valor em [0, 1].
    eye_wide_mean : float
        Média do blendshape ``eyeWide`` (L+R) na janela de 30 s.
        Valores baixos indicam olhos menos abertos.
    ear_trend : float
        Coeficiente angular (slope) linear do EAR na janela de 30 s.
        Valores negativos indicam olhos fechando progressivamente.
    """

    fatigue_level: str
    fatigue_score: float
    perclos: float
    eye_wide_mean: float
    ear_trend: float


class FatigueDetector:
    """
    Detector de fadiga ocular via PERCLOS, EAR temporal e blendshapes ARKit.

    Mantém buffers deslizantes de EAR, largura dos olhos e estados de
    fechamento para calcular, a cada frame, um indicador de fadiga baseado
    em três componentes com pesos distintos.

    Parâmetros
    ----------
    fps : float
        Taxa de quadros esperada do vídeo de entrada (padrão: 30.0).
        Usada para converter janelas temporais em número de frames.
    """

    def __init__(self, fps: float = 30.0):
        """
        Inicializa os buffers internos do detector.

        Parâmetros
        ----------
        fps : float
            Taxa de quadros estimada da fonte de vídeo.
        """
        self._fps            = fps
        self._perclos_frames = int(_PERCLOS_WINDOW_S * fps)
        self._ear_frames     = int(_EAR_WINDOW_S * fps)

        # Buffer (timestamp, ear) para tendência temporal
        self._ear_buf: deque[tuple[float, float]] = deque()
        # Buffer (eye_wide_L, eye_wide_R) para média de abertura
        self._wide_buf: deque[tuple[float, float]] = deque()
        # Buffer booleano: True = olho fechado ≥ 80%
        self._closure_buf: deque[bool] = deque()

    def _blendshape(self, face_data: FaceData, name: str) -> float:
        """
        Extrai o score de um blendshape específico pelo nome.

        Parâmetros
        ----------
        face_data : FaceData
            Dados da detecção facial com lista de blendshapes ARKit.
        name : str
            Nome do blendshape ARKit (ex.: ``'eyeWide_L'``).

        Retorna
        -------
        float
            Score do blendshape em [0, 1]; 0.0 se não encontrado.
        """
        if not face_data.blendshapes:
            return 0.0
        for bs in face_data.blendshapes:
            if bs.category_name == name:
                return float(bs.score)
        return 0.0

    def update(
        self, face_data: FaceData, blink_ear: float, timestamp: float
    ) -> Optional[FatigueResult]:
        """
        Atualiza o estado interno com o frame atual e retorna o resultado.

        Se nenhum rosto for detectado, retorna ``None`` sem alterar os buffers.

        Parâmetros
        ----------
        face_data : FaceData
            Dados de detecção facial do frame atual.
        blink_ear : float
            EAR médio calculado pelo BlinkDetector para o frame atual.
        timestamp : float
            Tempo em segundos desde o início da sessão.

        Retorna
        -------
        FatigueResult ou None
            Resultado com nível, score e métricas parciais, ou None se
            nenhum rosto foi detectado.
        """
        if not face_data.detected:
            return None

        eye_wide_l  = self._blendshape(face_data, "eyeWideLeft")
        eye_wide_r  = self._blendshape(face_data, "eyeWideRight")
        eye_blink_l = self._blendshape(face_data, "eyeBlinkLeft")
        eye_blink_r = self._blendshape(face_data, "eyeBlinkRight")
        avg_closure = (eye_blink_l + eye_blink_r) / 2.0

        self._ear_buf.append((timestamp, blink_ear))
        self._wide_buf.append((eye_wide_l, eye_wide_r))
        self._closure_buf.append(avg_closure >= _CLOSURE_THRESHOLD)

        # Trim buffers ao tamanho máximo
        while len(self._closure_buf) > self._perclos_frames:
            self._closure_buf.popleft()
        while len(self._ear_buf) > self._ear_frames:
            self._ear_buf.popleft()
        while len(self._wide_buf) > self._ear_frames:
            self._wide_buf.popleft()

        # ── PERCLOS ───────────────────────────────────────────────────────────
        perclos = sum(self._closure_buf) / len(self._closure_buf)

        # ── Slope do EAR (regressão linear) ──────────────────────────────────
        ear_trend = 0.0
        if len(self._ear_buf) >= 10:
            times = np.array([t for t, _ in self._ear_buf])
            ears  = np.array([e for _, e in self._ear_buf])
            if times[-1] > times[0]:
                ear_trend = float(np.polyfit(times - times[0], ears, 1)[0])

        # ── Média de abertura dos olhos ───────────────────────────────────────
        eye_wide_mean = float(np.mean([(l + r) / 2 for l, r in self._wide_buf]))

        # ── Score composto ────────────────────────────────────────────────────
        # PERCLOS: referência NHTSA — > 0.15 indica sonolência severa
        perclos_score = float(np.clip(perclos / 0.15, 0.0, 1.0))
        # Slope negativo de EAR: −0.005/s ou mais negativo = fadiga máxima
        trend_score   = float(np.clip(-ear_trend / 0.005, 0.0, 1.0))
        # eyeWide baixo: 0.3 = abertura típica alerta; abaixo = cansaço
        wide_score    = float(np.clip(1.0 - eye_wide_mean / 0.3, 0.0, 1.0))

        fatigue_score = float(np.clip(
            0.50 * perclos_score + 0.30 * trend_score + 0.20 * wide_score,
            0.0, 1.0,
        ))

        if fatigue_score < 0.25:
            level = "alert"
        elif fatigue_score < 0.50:
            level = "mild"
        elif fatigue_score < 0.75:
            level = "moderate"
        else:
            level = "severe"

        return FatigueResult(
            fatigue_level=level,
            fatigue_score=round(fatigue_score, 4),
            perclos=round(perclos, 4),
            eye_wide_mean=round(eye_wide_mean, 4),
            ear_trend=round(ear_trend, 6),
        )
