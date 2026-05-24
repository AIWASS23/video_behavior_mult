"""
Rastreamento de identidade de múltiplas pessoas via proximidade de centróide.

A cada frame, associa cada rosto detectado a um ID inteiro persistente usando
distância euclidiana entre centróides de landmarks. Novos IDs são criados para
rostos sem correspondência; tracks sem detecção por mais de ``max_missing_frames``
frames consecutivos são descartados.

Algoritmo
---------
1. Computar o centróide de cada rosto detectado (ponta do nariz, landmark 1).
2. Para cada detecção, encontrar o track existente mais próximo dentro de
   ``max_distance`` (coordenadas normalizadas).
3. Matching greedy: cada track só pode ser atribuído a uma detecção por frame.
4. Detecções sem match → novo ID (auto-incremento a partir de 0).
5. Tracks sem match → incrementar contador de ausências; descartar se exceder
   ``max_missing_frames``.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .face_tracker import FaceData


@dataclass
class _Track:
    """Estado interno de um track ativo."""
    center: tuple[float, float]
    missing: int = 0


class PersonTracker:
    """
    Rastreador de identidade de múltiplas pessoas por centróide de landmark.

    Mantém um dicionário de tracks ativos e, a cada chamada de ``update()``,
    associa as novas detecções aos tracks existentes pela menor distância
    euclidiana entre centróides em coordenadas normalizadas.

    Parâmetros
    ----------
    max_distance : float
        Distância máxima (coordenadas normalizadas [0,1]) para considerar
        que uma detecção corresponde a um track existente. Padrão: 0.20
        (≈ 20% da largura/altura do frame).
    max_missing_frames : int
        Número máximo de frames consecutivos sem detecção antes de um
        track ser descartado. Padrão: 30 (≈ 1 s a 30 fps).
    """

    def __init__(
        self,
        max_distance: float = 0.20,
        max_missing_frames: int = 30,
    ):
        """
        Inicializa o tracker sem tracks ativos.

        Parâmetros
        ----------
        max_distance : float
            Raio de associação em coordenadas normalizadas.
        max_missing_frames : int
            Tolerância de ausência antes de descartar um track.
        """
        self._max_distance     = max_distance
        self._max_missing      = max_missing_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id: int     = 0

    @property
    def active_ids(self) -> set[int]:
        """Conjunto dos IDs de pessoas atualmente rastreadas."""
        return set(self._tracks.keys())

    @property
    def person_count(self) -> int:
        """Número de pessoas rastreadas no frame mais recente."""
        return len(self._tracks)

    def _centroid(self, face_data: FaceData) -> tuple[float, float]:
        """
        Calcula o centróide de um rosto usando a ponta do nariz (landmark 1).

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais com landmarks normalizados.

        Retorna
        -------
        tuple[float, float]
            Par (x, y) em coordenadas normalizadas [0, 1].
        """
        lm = face_data.landmarks[1]
        return (float(lm.x), float(lm.y))

    def update(self, faces: list[FaceData]) -> list[tuple[int, FaceData]]:
        """
        Associa IDs persistentes à lista de rostos detectados no frame atual.

        Realiza matching greedy entre detecções e tracks existentes, cria
        novos IDs para detecções sem correspondência e descarta tracks
        ausentes por muitos frames.

        Parâmetros
        ----------
        faces : list[FaceData]
            Lista de rostos detectados no frame atual (saída do FaceTracker).

        Retorna
        -------
        list[tuple[int, FaceData]]
            Lista de pares ``(person_id, face_data)`` ordenada por
            ``person_id`` crescente.
        """
        if not faces:
            # Incrementa ausências de todos os tracks
            to_remove = []
            for tid, track in self._tracks.items():
                track.missing += 1
                if track.missing > self._max_missing:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._tracks[tid]
            return []

        centers = [self._centroid(fd) for fd in faces]
        assignments: dict[int, int] = {}   # det_idx → person_id
        used_tracks: set[int] = set()

        # Greedy matching: menor distância primeiro
        for det_idx, center in enumerate(centers):
            best_id:   Optional[int]   = None
            best_dist: float           = self._max_distance

            for tid, track in self._tracks.items():
                if tid in used_tracks:
                    continue
                dist = float(np.linalg.norm(
                    np.array(center) - np.array(track.center)
                ))
                if dist < best_dist:
                    best_dist = dist
                    best_id   = tid

            if best_id is not None:
                assignments[det_idx] = best_id
                used_tracks.add(best_id)
            else:
                assignments[det_idx] = self._next_id
                self._next_id += 1

        # Atualiza tracks com detecções associadas
        new_tracks: dict[int, _Track] = {}
        for det_idx, person_id in assignments.items():
            new_tracks[person_id] = _Track(center=centers[det_idx], missing=0)

        # Preserva tracks não associados (incrementa ausência)
        for tid, track in self._tracks.items():
            if tid not in new_tracks:
                track.missing += 1
                if track.missing <= self._max_missing:
                    new_tracks[tid] = track

        self._tracks = new_tracks

        result = [(assignments[i], faces[i]) for i in range(len(faces))]
        return sorted(result, key=lambda x: x[0])
