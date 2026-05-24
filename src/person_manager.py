"""
Gerenciador de instâncias de detectores por pessoa rastreada.

Para cada ``person_id`` novo, cria automaticamente um conjunto completo
de detectores independentes. Quando um ID sai de cena, os detectores
correspondentes são removidos, liberando memória e estado acumulado.

Esta abordagem permite reutilizar todos os detectores existentes sem
nenhuma modificação — cada pessoa tem seus próprios buffers de EAR,
sinal respiratório, histórico de blendshapes, etc.
"""


from .blink_detector import BlinkDetector
from .gaze_estimator import GazeEstimator
from .head_pose_estimator import HeadPoseEstimator
from .emotion_detector import EmotionDetector
from .breathing_detector import BreathingDetector
from .fatigue_detector import FatigueDetector
from .microexpression_detector import MicroexpressionDetector
from .asymmetry_detector import AsymmetryDetector


class PersonManager:
    """
    Gerencia um conjunto completo de detectores por pessoa rastreada.

    Instancia novos detectores na primeira aparição de um ``person_id``
    e remove os detectores quando o ID é declarado inativo via ``cleanup()``.

    Parâmetros
    ----------
    config : dict
        Dicionário de configuração carregado do YAML do projeto.
        Deve conter as seções ``blink``, ``gaze``, ``head_pose`` e ``emotion``.
    fps : float
        Taxa de quadros da fonte de vídeo. Usada pelos detectores que
        mantêm janelas temporais (fadiga, microexpressões).
    """

    def __init__(self, config: dict, fps: float = 30.0):
        """
        Inicializa o gerenciador sem nenhum conjunto de detectores ativo.

        Parâmetros
        ----------
        config : dict
            Configuração completa do projeto.
        fps : float
            Taxa de quadros estimada da fonte de vídeo.
        """
        self._config  = config
        self._fps     = fps
        self._persons: dict[int, dict] = {}

    def _create_detectors(self) -> dict:
        """
        Cria e retorna um novo conjunto de detectores para uma pessoa.

        Retorna
        -------
        dict
            Dicionário com as chaves: ``blinker``, ``gazer``, ``poser``,
            ``emoter``, ``breather``, ``fatiguer``, ``microer``, ``asymer``.
        """
        cfg = self._config
        return {
            "blinker": BlinkDetector(
                ear_threshold=cfg["blink"]["ear_threshold"],
                min_frames=cfg["blink"]["min_frames"],
                voluntary_min_ms=cfg["blink"]["voluntary_min_ms"],
            ),
            "gazer": GazeEstimator(
                h_threshold=cfg["gaze"]["horizontal_threshold"],
                v_threshold=cfg["gaze"]["vertical_threshold"],
            ),
            "poser": HeadPoseEstimator(
                frontal_threshold=cfg["head_pose"]["frontal_threshold"],
                movement_threshold=cfg["head_pose"]["sudden_movement_threshold"],
            ),
            "emoter": EmotionDetector(
                sample_interval=cfg["emotion"]["sample_interval"],
                enabled=cfg["emotion"]["enabled"],
            ),
            "breather":  BreathingDetector(),
            "fatiguer":  FatigueDetector(fps=self._fps),
            "microer":   MicroexpressionDetector(fps=self._fps),
            "asymer":    AsymmetryDetector(),
        }

    def get(self, person_id: int) -> dict:
        """
        Retorna os detectores da pessoa indicada, criando-os se necessário.

        Parâmetros
        ----------
        person_id : int
            ID persistente da pessoa, atribuído pelo PersonTracker.

        Retorna
        -------
        dict
            Conjunto de detectores para o ``person_id`` informado.
        """
        if person_id not in self._persons:
            self._persons[person_id] = self._create_detectors()
        return self._persons[person_id]

    def cleanup(self, active_ids: set[int]):
        """
        Remove os detectores de pessoas que não estão mais ativas.

        Deve ser chamado a cada frame após o PersonTracker atualizar os
        IDs ativos, para liberar memória de tracks descartados.

        Parâmetros
        ----------
        active_ids : set[int]
            Conjunto de IDs atualmente rastreados (retornado por
            ``PersonTracker.active_ids``).
        """
        stale = [pid for pid in self._persons if pid not in active_ids]
        for pid in stale:
            del self._persons[pid]

    @property
    def person_count(self) -> int:
        """Número de conjuntos de detectores ativos."""
        return len(self._persons)
