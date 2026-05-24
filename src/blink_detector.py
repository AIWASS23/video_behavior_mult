"""
Detecção de piscadas usando o método Eye Aspect Ratio (EAR).

Referência
----------
Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks",
CVWW 2016.

Fórmula do EAR
--------------
    EAR = (||P2-P6|| + ||P3-P5||) / (2 x ||P1-P4||)

Onde P1..P6 são seis landmarks do contorno do olho dispostos em sentido
horário: P1/P4 nas extremidades horizontais, P2/P3 e P5/P6 nas extremidades
verticais superior e inferior.

Classificação de piscadas
--------------------------
- Involuntária (reflexo): duração < voluntary_min_ms  (tipicamente 50–150 ms)
- Voluntária   (intencional): duração ≥ voluntary_min_ms

Índices dos landmarks do MediaPipe Face Mesh utilizados (6 pontos por olho):
  Olho esquerdo : [362, 385, 387, 263, 373, 380]  → P1..P6
  Olho direito  : [ 33, 160, 158, 133, 153, 144]  → P1..P6
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .face_tracker import FaceData


@dataclass
class BlinkEvent:
    """
    Representa um único evento de piscada detectado.

    Atributos
    ----------
    timestamp : float
        Momento de início da piscada em segundos a partir do início da sessão.
    duration_ms : float
        Duração total da piscada em milissegundos (do fechamento à reabertura).
    blink_type : str
        Classificação: ``'voluntary'`` (voluntária) ou ``'involuntary'`` (involuntária).
    left_ear : float
        Valor mínimo de EAR do olho esquerdo durante o evento.
    right_ear : float
        Valor mínimo de EAR do olho direito durante o evento.
    """

    timestamp: float
    duration_ms: float
    blink_type: str   # 'voluntary' | 'involuntary'
    left_ear: float
    right_ear: float


# ── Índices dos landmarks para cálculo do EAR (6 pontos por olho) ────────────
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [ 33, 160, 158, 133, 153, 144]


def _ear(landmarks, indices: list[int], w: int, h: int) -> float:
    """
    Calcula o Eye Aspect Ratio (EAR) para um olho.

    Extrai as coordenadas em pixels dos seis pontos de referência do olho
    e aplica a fórmula EAR = (||P2-P6|| + ||P3-P5||) / (2 x ||P1-P4||).

    Parâmetros
    ----------
    landmarks : NormalizedLandmarkList
        Lista de landmarks normalizados retornada pelo MediaPipe.
    indices : list[int]
        Seis índices de landmark na ordem [P1, P2, P3, P4, P5, P6].
    w : int
        Largura do frame em pixels (para desnormalização de coordenadas).
    h : int
        Altura do frame em pixels (para desnormalização de coordenadas).

    Retorna
    -------
    float
        Valor do EAR. Próximo de 0.0 indica olho fechado;
        valores normais em repouso ficam entre 0.25 e 0.40.
        Retorna 1.0 se a abertura horizontal for nula (evita divisão por zero).
    """
    pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in indices]
    )
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 1.0


class BlinkDetector:
    """
    Detector de piscadas baseado em máquina de estados com o método EAR.

    Monitora frame a frame se o EAR médio cai abaixo de um limiar por um
    número mínimo de frames consecutivos. Ao reabertura do olho, classifica
    a piscada como voluntária ou involuntária com base na duração total.

    Parâmetros
    ----------
    ear_threshold : float
        Valor de EAR abaixo do qual o olho é considerado fechado (padrão: 0.22).
        Pode precisar de ajuste individual: usuários com olhos naturalmente
        estreitos podem exigir valores menores.
    min_frames : int
        Número mínimo de frames consecutivos com EAR abaixo do limiar para
        que o evento seja registrado como piscada válida (padrão: 2).
        Evita falsos positivos por ruído momentâneo nos landmarks.
    voluntary_min_ms : float
        Duração mínima em ms para classificar uma piscada como voluntária
        (padrão: 150 ms). Piscadas reflexas normais duram ~100–150 ms.
    """

    def __init__(
        self,
        ear_threshold: float = 0.22,
        min_frames: int = 2,
        voluntary_min_ms: float = 150.0,
    ):
        """
        Inicializa o detector e reseta o estado interno da máquina de estados.

        Parâmetros
        ----------
        ear_threshold : float, opcional
            Limiar de EAR para detecção de olho fechado (padrão: 0.22).
        min_frames : int, opcional
            Frames consecutivos mínimos para validar a piscada (padrão: 2).
        voluntary_min_ms : float, opcional
            Duração mínima em ms para piscada voluntária (padrão: 150.0).
        """
        self.ear_threshold    = ear_threshold
        self.min_frames       = min_frames
        self.voluntary_min_ms = voluntary_min_ms

        self.blinks: list[BlinkEvent] = []  # histórico completo da sessão
        self.current_ear: float = 1.0       # EAR médio do frame atual

        # ── estado interno da máquina de estados ──────────────────────────
        self._in_blink    = False
        self._blink_start: Optional[float] = None
        self._blink_frames = 0
        self._peak_left    = 1.0  # EAR mínimo esquerdo durante o evento
        self._peak_right   = 1.0  # EAR mínimo direito durante o evento

    def update(self, face_data: FaceData, timestamp: float) -> Optional[BlinkEvent]:
        """
        Atualiza a máquina de estados com os dados do frame atual.

        Deve ser chamado a cada frame processado, na ordem cronológica.
        Retorna um BlinkEvent apenas no frame em que o olho reabre após
        uma sequência fechada suficientemente longa.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais do frame atual (landmarks do MediaPipe).
        timestamp : float
            Tempo em segundos desde o início da sessão.

        Retorna
        -------
        BlinkEvent ou None
            BlinkEvent preenchido quando uma piscada completa é detectada
            (no frame de reabertura); None em todos os outros casos.
        """
        if not face_data.detected:
            self._reset()
            return None

        lm = face_data.landmarks
        w, h = face_data.frame_w, face_data.frame_h

        left  = _ear(lm, _LEFT_EYE,  w, h)
        right = _ear(lm, _RIGHT_EYE, w, h)
        avg   = (left + right) / 2.0
        self.current_ear = avg

        # ── olho fechado: acumula estado ──────────────────────────────────
        if avg < self.ear_threshold:
            if not self._in_blink:
                self._in_blink     = True
                self._blink_start  = timestamp
                self._blink_frames = 1
                self._peak_left    = left
                self._peak_right   = right
            else:
                self._blink_frames += 1
                self._peak_left    = min(self._peak_left,  left)
                self._peak_right   = min(self._peak_right, right)
            return None

        # ── olho reaberto: verifica se o fechamento foi longo o suficiente ─
        if self._in_blink and self._blink_frames >= self.min_frames:
            duration_ms = (timestamp - self._blink_start) * 1000.0
            blink_type  = (
                "voluntary" if duration_ms >= self.voluntary_min_ms else "involuntary"
            )
            event = BlinkEvent(
                timestamp=self._blink_start,
                duration_ms=duration_ms,
                blink_type=blink_type,
                left_ear=self._peak_left,
                right_ear=self._peak_right,
            )
            self.blinks.append(event)
            self._reset()
            return event

        self._reset()
        return None

    def get_blink_rate(self, elapsed_seconds: float) -> float:
        """
        Calcula a taxa de piscadas por minuto para a sessão atual.

        Parâmetros
        ----------
        elapsed_seconds : float
            Tempo total decorrido desde o início da sessão em segundos.

        Retorna
        -------
        float
            Piscadas por minuto. Retorna 0.0 se menos de 1 segundo tiver
            decorrido (evita taxa inflacionada no início da sessão).
        """
        if elapsed_seconds < 1.0:
            return 0.0
        return len(self.blinks) / elapsed_seconds * 60.0

    def get_stats(self) -> dict:
        """
        Retorna um dicionário com estatísticas agregadas de piscadas.

        Inclui contagens totais, por tipo e estatísticas de duração.

        Retorna
        -------
        dict
            Dicionário com as chaves:
            - ``total_blinks``    : int   — total de piscadas registradas
            - ``voluntary``       : int   — piscadas voluntárias
            - ``involuntary``     : int   — piscadas involuntárias
            - ``avg_duration_ms`` : float — duração média em ms
            - ``min_duration_ms`` : float — menor duração registrada em ms
            - ``max_duration_ms`` : float — maior duração registrada em ms
        """
        total     = len(self.blinks)
        voluntary = sum(1 for b in self.blinks if b.blink_type == "voluntary")
        durations = [b.duration_ms for b in self.blinks]
        return {
            "total_blinks":    total,
            "voluntary":       voluntary,
            "involuntary":     total - voluntary,
            "avg_duration_ms": float(np.mean(durations)) if durations else 0.0,
            "min_duration_ms": float(np.min(durations))  if durations else 0.0,
            "max_duration_ms": float(np.max(durations))  if durations else 0.0,
        }

    def _reset(self):
        """
        Reseta o estado interno da máquina de estados para o valor inicial.

        Chamado internamente quando o olho reabre (com ou sem piscada válida)
        ou quando nenhum rosto é detectado no frame.
        """
        self._in_blink     = False
        self._blink_start  = None
        self._blink_frames = 0
        self._peak_left    = 1.0
        self._peak_right   = 1.0
