"""
Estimativa de pose da cabeça via PnP (Perspective-n-Point).

Seis landmarks faciais estáveis são combinados com um modelo 3D genérico do
rosto humano (em milímetros, com a ponta do nariz como origem). O método
cv2.solvePnP resolve a rotação que melhor alinha o modelo 3D com os pontos
2D detectados no frame, retornando um vetor de rotação que é convertido em
ângulos de Euler: pitch (inclinação vertical), yaw (rotação horizontal) e
roll (inclinação lateral).

Aproximação da câmera
---------------------
Os parâmetros intrínsecos são estimados como um modelo pinhole simples
derivado das dimensões do frame. Esta aproximação é suficiente para
classificação angular em webcam convencional sem calibração formal.

Índices dos landmarks do MediaPipe utilizados
----------------------------------------------
  1   → ponta do nariz  (origem do modelo 3D)
  152 → queixo
  33  → canto externo do olho direito
  263 → canto externo do olho esquerdo
  61  → canto direito da boca
  291 → canto esquerdo da boca
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .face_tracker import FaceData


@dataclass
class HeadPose:
    """
    Ângulos de Euler e flags de estado para a pose da cabeça em um frame.

    Atributos
    ----------
    yaw : float
        Rotação em torno do eixo Y em graus.
        Valores positivos indicam face virada para a direita no espaço de imagem.
    pitch : float
        Rotação em torno do eixo X em graus.
        Valores positivos indicam face inclinada para baixo.
    roll : float
        Rotação em torno do eixo Z em graus (inclinação lateral da cabeça).
    is_frontal : bool
        True quando |yaw| e |pitch| estão ambos abaixo do limiar frontal
        configurado, indicando que o usuário olha aproximadamente para a câmera.
    sudden_movement : bool
        True quando a variação angular em relação ao frame anterior excede
        o limiar de movimento brusco configurado.
    """

    yaw: float
    pitch: float
    roll: float
    is_frontal: bool
    sudden_movement: bool


# ── Modelo 3D canônico do rosto (mm, sistema destro, nariz = origem) ──────────
_MODEL_3D = np.array([
    (  0.0,   0.0,   0.0),   # 1   ponta do nariz
    (  0.0, -63.6, -12.5),   # 152 queixo
    (-43.3,  32.7, -26.0),   # 33  canto externo do olho direito
    ( 43.3,  32.7, -26.0),   # 263 canto externo do olho esquerdo
    (-28.9, -28.9, -24.1),   # 61  canto direito da boca
    ( 28.9, -28.9, -24.1),   # 291 canto esquerdo da boca
], dtype=np.float64)

_LM_IDS = [1, 152, 33, 263, 61, 291]
# ─────────────────────────────────────────────────────────────────────────────


def _rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    Converte uma matriz de rotação 3×3 para ângulos de Euler (pitch, yaw, roll).

    Usa a decomposição ZYX (rotação extrínseca). Detecta e trata o caso de
    gimbal lock (singularidade) quando a norma do bloco superior esquerdo
    é próxima de zero.

    Parâmetros
    ----------
    R : np.ndarray
        Matriz de rotação de forma (3, 3).

    Retorna
    -------
    tuple[float, float, float]
        Tripla (pitch, yaw, roll) em graus.
        - pitch: inclinação vertical (rotação em X)
        - yaw:   rotação horizontal (rotação em Y)
        - roll:  inclinação lateral (rotação em Z)
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        # Singularidade (gimbal lock): roll fixo em zero
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = 0.0
    return pitch, yaw, roll


class HeadPoseEstimator:
    """
    Estimador de pose 3D da cabeça usando solvePnP do OpenCV.

    A cada frame, seis pontos 2D detectados pelo MediaPipe são alinhados ao
    modelo 3D canônico via solução PnP iterativa (SOLVEPNP_ITERATIVE), obtendo
    a matriz de rotação que é decomposta em ângulos de Euler. A variação angular
    entre frames consecutivos permite detectar movimentos bruscos de cabeça.

    Parâmetros
    ----------
    frontal_threshold : float
        Limiar em graus (absoluto) para |yaw| e |pitch|. Quando ambos estão
        abaixo desse valor, a pose é considerada frontal (padrão: 20.0°).
    movement_threshold : float
        Variação máxima em graus (por frame) no yaw ou pitch antes de acionar
        o flag de movimento brusco (padrão: 15.0°).
    """

    def __init__(
        self,
        frontal_threshold: float = 20.0,
        movement_threshold: float = 15.0,
    ):
        """
        Inicializa o estimador de pose com os limiares de classificação.

        Parâmetros
        ----------
        frontal_threshold : float, opcional
            Limiar angular em graus para classificação como postura frontal
            (padrão: 20.0).
        movement_threshold : float, opcional
            Delta angular em graus/frame para flagrar movimento brusco
            (padrão: 15.0).
        """
        self.frontal_threshold  = frontal_threshold
        self.movement_threshold = movement_threshold
        self._prev: Optional[HeadPose] = None  # pose do frame anterior

    def update(
        self,
        face_data: FaceData,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> Optional[HeadPose]:
        """
        Estima a pose da cabeça para o frame atual.

        Extrai os seis landmarks 2D do FaceData, resolve o problema PnP com
        o modelo 3D canônico e converte o vetor de rotação resultante em
        ângulos de Euler. Compara com o frame anterior para detectar
        movimentos bruscos.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais do frame atual com landmarks do MediaPipe.
        camera_matrix : np.ndarray ou None, opcional
            Matriz de câmera 3×3. Se None, uma aproximação pinhole é derivada
            automaticamente a partir das dimensões do frame:
            focal_length = largura_do_frame, ponto_principal = centro do frame.

        Retorna
        -------
        HeadPose ou None
            Objeto HeadPose com ângulos e flags calculados, ou None se nenhum
            rosto for detectado ou se o solvePnP falhar.
        """
        if not face_data.detected:
            self._prev = None
            return None

        h, w = face_data.frame_h, face_data.frame_w

        if camera_matrix is None:
            f = float(w)
            camera_matrix = np.array(
                [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64
            )

        # Coleta os seis pontos 2D correspondentes ao modelo 3D
        pts_2d = np.array(
            [
                [face_data.landmarks[i].x * w,
                 face_data.landmarks[i].y * h]
                for i in _LM_IDS
            ],
            dtype=np.float64,
        )

        dist = np.zeros((4, 1))  # coeficientes de distorção assumidos nulos
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_3D, pts_2d, camera_matrix, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_to_euler(rmat)

        is_frontal = (
            abs(yaw)   < self.frontal_threshold and
            abs(pitch) < self.frontal_threshold
        )

        sudden = False
        if self._prev is not None:
            delta  = max(abs(yaw - self._prev.yaw), abs(pitch - self._prev.pitch))
            sudden = delta > self.movement_threshold

        pose = HeadPose(
            yaw=yaw, pitch=pitch, roll=roll,
            is_frontal=is_frontal, sudden_movement=sudden,
        )
        self._prev = pose
        return pose
