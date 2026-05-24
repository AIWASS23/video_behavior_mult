"""
Módulo de rastreamento facial baseado no MediaPipe Face Landmarker (Tasks API).

A partir do MediaPipe 0.10, a API legada ``mp.solutions`` foi removida.
Este módulo usa a nova Tasks API (``mediapipe.tasks.python.vision``), que
requer um arquivo de modelo ``.task`` baixado separadamente via
``download_models.py``.

O modelo ``face_landmarker.task`` produz 478 landmarks por rosto:
  - 468 landmarks base do contorno facial
  - 10 landmarks extras das íris (468–477), necessários para a estimativa
    de direção do olhar no GazeEstimator.
"""

import os

import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Optional

# Caminho padrão do modelo (relativo à raiz do projeto)
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "face_landmarker.task"
)


@dataclass
class FaceData:
    """
    Contêiner imutável com os dados de detecção facial de um único frame.

    Na nova Tasks API, ``landmarks`` é uma lista Python de objetos
    ``NormalizedLandmark``, acessados diretamente por índice:
    ``landmarks[idx].x``, ``landmarks[idx].y``.

    Atributos
    ----------
    detected : bool
        True se pelo menos um rosto foi detectado no frame.
    landmarks : list ou None
        Lista de 478 objetos NormalizedLandmark do MediaPipe.
        None quando detected=False.
    blendshapes : list ou None
        Lista de objetos Category com os 52 blendshapes faciais (ARKit),
        cada um com atributos ``.category_name`` (str) e ``.score`` (float em [0,1]).
        Usada pelo EmotionDetector para estimar emoções sem TensorFlow.
        None quando detected=False.
    frame_h : int
        Altura do frame em pixels.
    frame_w : int
        Largura do frame em pixels.
    """

    detected: bool
    landmarks: Optional[list]    # list[NormalizedLandmark], 478 elementos
    blendshapes: Optional[list]  # list[Category], 52 blendshapes ARKit
    frame_h: int
    frame_w: int

    def get_landmark_px(self, idx: int) -> tuple[int, int]:
        """
        Retorna as coordenadas em pixels de um landmark.

        Converte as coordenadas normalizadas [0, 1] do MediaPipe para
        coordenadas absolutas no espaço do frame.

        Parâmetros
        ----------
        idx : int
            Índice do landmark (0–477).

        Retorna
        -------
        tuple[int, int]
            Par (x, y) em pixels.
        """
        lm = self.landmarks[idx]
        return (int(lm.x * self.frame_w), int(lm.y * self.frame_h))

    def get_landmark_norm(self, idx: int) -> tuple[float, float]:
        """
        Retorna as coordenadas normalizadas [0, 1] de um landmark.

        Parâmetros
        ----------
        idx : int
            Índice do landmark (0–477).

        Retorna
        -------
        tuple[float, float]
            Par (x, y) no espaço normalizado.
        """
        lm = self.landmarks[idx]
        return (lm.x, lm.y)

    def get_landmark_np(self, idx: int) -> np.ndarray:
        """
        Retorna as coordenadas normalizadas de um landmark como array NumPy.

        Útil para operações vetoriais (distâncias, razões, PnP).

        Parâmetros
        ----------
        idx : int
            Índice do landmark (0–477).

        Retorna
        -------
        np.ndarray
            Array de forma (2,) com [x, y] normalizados.
        """
        lm = self.landmarks[idx]
        return np.array([lm.x, lm.y])


class FaceTracker:
    """
    Rastreador facial baseado no MediaPipe Face Landmarker (Tasks API).

    Gerencia o ciclo de vida do modelo e processa frames RGB produzindo
    uma lista de objetos FaceData — um por rosto detectado. Suporta
    múltiplos rostos simultâneos (configurável via ``max_faces``).

    Parâmetros
    ----------
    model_path : str
        Caminho para o arquivo ``face_landmarker.task``. Se None, usa o
        caminho padrão ``models/face_landmarker.task`` relativo à raiz
        do projeto.
    max_faces : int
        Número máximo de rostos a detectar por frame (padrão: 10).
    min_detection_confidence : float
        Confiança mínima para a detecção inicial do rosto (0–1).
    min_tracking_confidence : float
        Confiança mínima para rastreamento entre frames consecutivos (0–1).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_faces: int = 10,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Inicializa e carrega o modelo MediaPipe Face Landmarker.

        Parâmetros
        ----------
        model_path : str ou None, opcional
            Caminho para o arquivo .task. None usa o padrão do projeto.
        min_detection_confidence : float, opcional
            Confiança mínima de detecção (padrão: 0.5).
        min_tracking_confidence : float, opcional
            Confiança mínima de rastreamento (padrão: 0.5).

        Levanta
        -------
        FileNotFoundError
            Se o arquivo de modelo não for encontrado. Execute
            ``python download_models.py`` para baixar o modelo.
        """
        path = model_path or os.path.abspath(_MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {path}\n"
                "Execute  python download_models.py  para baixar o modelo."
            )

        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=True,   # habilita os 52 blendshapes ARKit
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_rgb: np.ndarray) -> list[FaceData]:
        """
        Processa um frame RGB e retorna a lista de rostos detectados.

        O frame deve estar no formato RGB (não BGR). Use
        ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`` antes de chamar este método.

        Parâmetros
        ----------
        frame_rgb : np.ndarray
            Frame de entrada no formato HxWx3 RGB, dtype uint8.

        Retorna
        -------
        list[FaceData]
            Lista com um FaceData por rosto detectado, na ordem retornada
            pelo MediaPipe. Retorna lista vazia se nenhum rosto for
            encontrado.
        """
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result   = self._detector.detect(mp_image)

        faces = []
        for i, landmarks in enumerate(result.face_landmarks):
            blendshapes = (
                result.face_blendshapes[i]
                if result.face_blendshapes and i < len(result.face_blendshapes)
                else None
            )
            faces.append(FaceData(
                detected=True,
                landmarks=landmarks,
                blendshapes=blendshapes,
                frame_h=h,
                frame_w=w,
            ))
        return faces

    def close(self):
        """
        Libera os recursos do modelo MediaPipe.

        Deve ser chamado ao final do processamento para fechar corretamente
        os recursos internos do detector.
        """
        self._detector.close()
