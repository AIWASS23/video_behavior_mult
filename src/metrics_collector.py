"""
Coleta e agregação de métricas visuais por frame.

O MetricsCollector recebe as saídas de todos os módulos de análise a cada
frame e as armazena em uma lista de registros tipados. Ao final da sessão,
disponibiliza o histórico completo como DataFrame pandas (para exportação CSV)
e um dicionário de sumário com as estatísticas mais relevantes.
"""


import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .face_tracker import FaceData
from .blink_detector import BlinkEvent
from .gaze_estimator import GazeResult
from .head_pose_estimator import HeadPose
from .emotion_detector import EmotionResult
from .breathing_detector import BreathingResult
from .fatigue_detector import FatigueResult
from .microexpression_detector import MicroexpressionResult
from .asymmetry_detector import AsymmetryResult


@dataclass
class FrameRecord:
    """
    Registro de todas as métricas extraídas em um único frame.

    Cada instância corresponde a um frame processado e armazena os valores
    de todos os módulos de análise naquele instante de tempo. Esta estrutura
    serve como unidade de persistência no CSV de saída.

    Atributos
    ----------
    timestamp : float
        Tempo em segundos desde o início da sessão.
    frame_idx : int
        Índice sequencial do frame (começa em 0).
    face_detected : bool
        True se um rosto foi detectado neste frame.
    blink_occurred : bool
        True se uma piscada completa foi registrada neste frame.
    blink_type : str
        ``'voluntary'``, ``'involuntary'`` ou ``''`` (sem piscada).
    ear : float
        Eye Aspect Ratio médio dos dois olhos neste frame.
    gaze_direction : str
        Direção do olhar estimada (ex.: ``'center'``, ``'left'``, ``'up-right'``).
    gaze_h : float
        Razão horizontal normalizada da posição da íris [0, 1].
    gaze_v : float
        Razão vertical normalizada da posição da íris [0, 1].
    on_screen : bool
        True quando a heurística indica atenção à tela.
    head_yaw : float
        Ângulo de yaw da cabeça em graus.
    head_pitch : float
        Ângulo de pitch da cabeça em graus.
    head_roll : float
        Ângulo de roll da cabeça em graus.
    head_frontal : bool
        True quando a pose é classificada como frontal.
    sudden_movement : bool
        True quando um movimento brusco de cabeça foi detectado.
    dominant_emotion : str
        Emoção dominante (ou ``'unknown'`` se não disponível).
    emotion_scores : str
        Dicionário JSON com pontuação de cada emoção. Armazenado como string
        para manter o DataFrame plano e compatível com CSV.
    """

    timestamp: float
    frame_idx: int
    person_id: int
    face_detected: bool
    # ── Piscadas ──────────────────────────────────────────────────────────────
    blink_occurred: bool
    blink_type: str
    ear: float
    # ── Direção do olhar ──────────────────────────────────────────────────────
    gaze_direction: str
    gaze_h: float
    gaze_v: float
    on_screen: bool
    # ── Pose da cabeça ────────────────────────────────────────────────────────
    head_yaw: float
    head_pitch: float
    head_roll: float
    head_frontal: bool
    sudden_movement: bool
    # ── Emoções ───────────────────────────────────────────────────────────────
    dominant_emotion: str
    emotion_scores: str  # JSON serializado para compatibilidade com CSV
    # ── Respiração ───────────────────────────────────────────────────────────
    respiratory_rate: float
    breath_depth: str
    breath_phase: str
    breath_amplitude: float
    breath_regularity: float
    nostril_width_norm: float
    # ── Fadiga ocular ─────────────────────────────────────────────────────────
    fatigue_level: str
    fatigue_score: float
    perclos: float
    ear_trend: float
    # ── Microexpressões ───────────────────────────────────────────────────────
    microexp_detected: bool
    microexp_emotion: str
    microexp_intensity: float
    microexp_duration_ms: float
    # ── Assimetria facial ─────────────────────────────────────────────────────
    asymmetry_score: float
    asymmetry_dominant_side: str
    asymmetry_region_scores: str  # JSON serializado


class MetricsCollector:
    """
    Acumulador de métricas visuais por frame com exportação e sumário.

    Coleta os resultados dos módulos de análise frame a frame e disponibiliza
    o histórico completo como DataFrame pandas e um dicionário de estatísticas
    agregadas ao final da sessão.

    Uso típico
    ----------
    Instanciar uma vez antes do loop principal e chamar ``update()`` a cada
    frame. Ao encerrar a sessão, chamar ``save_csv()`` e ``get_summary()``.
    """

    def __init__(self):
        """Inicializa o coletor com lista de registros vazia."""
        self.records: list[FrameRecord] = []

    def update(
        self,
        timestamp: float,
        frame_idx: int,
        person_id: int,
        face_data: FaceData,
        blink_event: Optional[BlinkEvent],
        blink_ear: float,
        gaze_result: Optional[GazeResult],
        head_pose: Optional[HeadPose],
        emotion_result: Optional[EmotionResult],
        breathing_result: Optional[BreathingResult] = None,
        fatigue_result: Optional[FatigueResult] = None,
        micro_result: Optional[MicroexpressionResult] = None,
        asym_result: Optional[AsymmetryResult] = None,
    ) -> FrameRecord:
        """
        Registra as métricas de um frame e retorna o registro criado.

        Aceita ``None`` em qualquer parâmetro opcional (indicando módulo
        desabilitado ou ausência de detecção) e preenche com valores neutros.

        Parâmetros
        ----------
        timestamp : float
            Tempo em segundos desde o início da sessão.
        frame_idx : int
            Índice sequencial do frame.
        face_data : FaceData
            Resultado da detecção facial para o frame atual.
        blink_event : BlinkEvent ou None
            Evento de piscada (não None apenas no frame de conclusão da piscada).
        blink_ear : float
            EAR médio atual, independente de ter ocorrido piscada.
        gaze_result : GazeResult ou None
            Resultado da estimativa de olhar, ou None se indisponível.
        head_pose : HeadPose ou None
            Resultado da estimativa de pose, ou None se indisponível.
        emotion_result : EmotionResult ou None
            Resultado da classificação de emoções, ou None se indisponível.

        Retorna
        -------
        FrameRecord
            Registro criado e adicionado ao histórico interno.
        """
        rec = FrameRecord(
            timestamp=timestamp,
            frame_idx=frame_idx,
            person_id=person_id,
            face_detected=face_data.detected,
            # piscadas
            blink_occurred=blink_event is not None,
            blink_type=blink_event.blink_type if blink_event else "",
            ear=blink_ear,
            # olhar
            gaze_direction=gaze_result.direction       if gaze_result else "unknown",
            gaze_h=gaze_result.horizontal_ratio         if gaze_result else 0.5,
            gaze_v=gaze_result.vertical_ratio           if gaze_result else 0.5,
            on_screen=gaze_result.on_screen              if gaze_result else False,
            # pose da cabeça
            head_yaw=head_pose.yaw                      if head_pose else 0.0,
            head_pitch=head_pose.pitch                  if head_pose else 0.0,
            head_roll=head_pose.roll                    if head_pose else 0.0,
            head_frontal=head_pose.is_frontal           if head_pose else False,
            sudden_movement=head_pose.sudden_movement   if head_pose else False,
            # emoções
            dominant_emotion=emotion_result.dominant    if emotion_result else "unknown",
            emotion_scores=(
                json.dumps(emotion_result.scores) if emotion_result else "{}"
            ),
            # respiração
            respiratory_rate=breathing_result.respiratory_rate   if breathing_result else 0.0,
            breath_depth=breathing_result.breath_depth           if breathing_result else "unknown",
            breath_phase=breathing_result.breath_phase           if breathing_result else "hold",
            breath_amplitude=breathing_result.signal_amplitude   if breathing_result else 0.0,
            breath_regularity=breathing_result.regularity        if breathing_result else 0.0,
            nostril_width_norm=breathing_result.nostril_width_norm if breathing_result else 0.0,
            # fadiga
            fatigue_level=fatigue_result.fatigue_level   if fatigue_result else "unknown",
            fatigue_score=fatigue_result.fatigue_score   if fatigue_result else 0.0,
            perclos=fatigue_result.perclos               if fatigue_result else 0.0,
            ear_trend=fatigue_result.ear_trend           if fatigue_result else 0.0,
            # microexpressões
            microexp_detected=micro_result.detected                             if micro_result else False,
            microexp_emotion=micro_result.event.emotion if (micro_result and micro_result.event) else "",
            microexp_intensity=micro_result.event.intensity if (micro_result and micro_result.event) else 0.0,
            microexp_duration_ms=micro_result.event.duration_ms if (micro_result and micro_result.event) else 0.0,
            # assimetria
            asymmetry_score=asym_result.asymmetry_score             if asym_result else 0.0,
            asymmetry_dominant_side=asym_result.dominant_side       if asym_result else "unknown",
            asymmetry_region_scores=(
                json.dumps(asym_result.region_scores) if asym_result else "{}"
            ),
        )
        self.records.append(rec)
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte o histórico completo de registros em um DataFrame pandas.

        Cada linha corresponde a um frame e cada coluna a um atributo de
        FrameRecord. Útil para análise exploratória posterior e para a
        geração de gráficos no Visualizer.

        Retorna
        -------
        pd.DataFrame
            DataFrame com uma linha por frame e as colunas documentadas em
            FrameRecord. DataFrame vazio se nenhum frame foi registrado.
        """
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_csv(self, path: str):
        """
        Exporta o histórico completo de métricas para um arquivo CSV.

        Parâmetros
        ----------
        path : str
            Caminho completo do arquivo de saída (incluindo extensão .csv).
            O diretório pai deve existir; o arquivo é sobrescrito se já existir.
        """
        self.to_dataframe().to_csv(path, index=False)
        print(f"[Metrics] CSV salvo → {path}")

    def get_summary(self, elapsed_seconds: float) -> dict:
        """
        Calcula e retorna as estatísticas agregadas da sessão.

        Consolida todas as métricas coletadas em um dicionário com indicadores
        de alto nível adequados para exibição em relatório ou terminal.

        Parâmetros
        ----------
        elapsed_seconds : float
            Duração total da sessão em segundos. Usado para calcular taxas
            normalizadas por minuto e percentuais de estabilidade.

        Retorna
        -------
        dict
            Dicionário com as seguintes chaves:

            - ``duration_seconds``    : float — duração total da sessão
            - ``total_frames``        : int   — total de frames processados
            - ``face_stability_pct``  : float — % de frames com rosto detectado
            - ``total_blinks``        : int   — total de piscadas
            - ``voluntary_blinks``    : int   — piscadas voluntárias
            - ``involuntary_blinks``  : int   — piscadas involuntárias
            - ``blink_rate_per_min``  : float — taxa de piscadas por minuto
            - ``on_screen_pct``       : float — % do tempo com atenção à tela
            - ``gaze_distribution``   : dict  — contagem de frames por direção do olhar
            - ``dominant_emotion``    : str   — emoção mais frequente na sessão
            - ``emotion_distribution``: dict  — contagem de frames por emoção
            - ``sudden_movements``    : int   — total de movimentos bruscos detectados
            - ``avg_head_yaw_deg``    : float — yaw médio absoluto em graus
            - ``avg_head_pitch_deg``  : float — pitch médio absoluto em graus

            Retorna dicionário vazio se nenhum frame foi registrado.
        """
        if not self.records:
            return {}

        df  = self.to_dataframe()
        det = df[df["face_detected"]]

        blinks    = df[df["blink_occurred"]]
        total_b   = len(blinks)
        voluntary = int((blinks["blink_type"] == "voluntary").sum())

        blink_rate    = total_b / elapsed_seconds * 60.0 if elapsed_seconds > 0 else 0.0
        on_screen_pct = float(det["on_screen"].mean() * 100) if len(det) > 0 else 0.0

        gaze_counts    = det["gaze_direction"].value_counts().to_dict() if len(det) > 0 else {}
        emotion_counts = det["dominant_emotion"].value_counts().to_dict() if len(det) > 0 else {}

        sudden_moves = int(det["sudden_movement"].sum()) if len(det) > 0 else 0
        avg_yaw      = float(det["head_yaw"].abs().mean())   if len(det) > 0 else 0.0
        avg_pitch    = float(det["head_pitch"].abs().mean())  if len(det) > 0 else 0.0

        face_stability = float(len(det) / len(df) * 100) if len(df) > 0 else 0.0

        # Emoção dominante: ignora 'unknown' se houver outras disponíveis
        dominant_emotion = next(iter(emotion_counts), "unknown")

        # ── Métricas de respiração ─────────────────────────────────────────
        breath_det = det[det["respiratory_rate"] > 0]
        avg_resp_rate   = float(breath_det["respiratory_rate"].mean()) if len(breath_det) > 0 else 0.0
        avg_breath_reg  = float(det["breath_regularity"].mean())       if len(det) > 0 else 0.0
        breath_depth_counts = det["breath_depth"].value_counts().to_dict() if len(det) > 0 else {}
        breath_phase_counts = det["breath_phase"].value_counts().to_dict() if len(det) > 0 else {}

        # ── Métricas de fadiga ─────────────────────────────────────────────────
        fatigue_det = det[det["fatigue_level"] != "unknown"]
        avg_fatigue_score   = float(fatigue_det["fatigue_score"].mean()) if len(fatigue_det) > 0 else 0.0
        avg_perclos         = float(fatigue_det["perclos"].mean())        if len(fatigue_det) > 0 else 0.0
        fatigue_level_dist  = fatigue_det["fatigue_level"].value_counts().to_dict() if len(fatigue_det) > 0 else {}
        dominant_fatigue    = next(iter(fatigue_det["fatigue_level"].value_counts().index), "unknown") if len(fatigue_det) > 0 else "unknown"

        # ── Métricas de microexpressões ────────────────────────────────────────
        micros              = det[det["microexp_detected"]]
        total_micros        = len(micros)
        micro_emotion_dist  = micros["microexp_emotion"].value_counts().to_dict() if total_micros > 0 else {}
        avg_micro_intensity = float(micros["microexp_intensity"].mean())           if total_micros > 0 else 0.0

        # ── Métricas de assimetria ─────────────────────────────────────────────
        asym_det          = det[det["asymmetry_score"] > 0]
        avg_asymmetry     = float(asym_det["asymmetry_score"].mean())              if len(asym_det) > 0 else 0.0
        dominant_side_dist = det["asymmetry_dominant_side"].value_counts().to_dict() if len(det) > 0 else {}

        person_ids    = sorted(df["person_id"].unique().tolist())
        person_count  = len(person_ids)

        return {
            "duration_seconds":      elapsed_seconds,
            "total_frames":          len(df),
            "person_count":          person_count,
            "person_ids":            person_ids,
            "face_stability_pct":    face_stability,
            "total_blinks":          total_b,
            "voluntary_blinks":      voluntary,
            "involuntary_blinks":    total_b - voluntary,
            "blink_rate_per_min":    blink_rate,
            "on_screen_pct":         on_screen_pct,
            "gaze_distribution":     gaze_counts,
            "dominant_emotion":      dominant_emotion,
            "emotion_distribution":  emotion_counts,
            "sudden_movements":      sudden_moves,
            "avg_head_yaw_deg":      avg_yaw,
            "avg_head_pitch_deg":    avg_pitch,
            "avg_respiratory_rate":  avg_resp_rate,
            "avg_breath_regularity": avg_breath_reg,
            "breath_depth_dist":     breath_depth_counts,
            "breath_phase_dist":     breath_phase_counts,
            # fadiga
            "avg_fatigue_score":     avg_fatigue_score,
            "avg_perclos":           avg_perclos,
            "dominant_fatigue":      dominant_fatigue,
            "fatigue_level_dist":    fatigue_level_dist,
            # microexpressões
            "total_microexpressions":    total_micros,
            "avg_micro_intensity":       avg_micro_intensity,
            "microexpression_dist":      micro_emotion_dist,
            # assimetria
            "avg_asymmetry_score":   avg_asymmetry,
            "dominant_side_dist":    dominant_side_dist,
        }

    def get_summary_per_person(self, elapsed_seconds: float) -> dict[int, dict]:
        """
        Calcula e retorna estatísticas agregadas separadas por pessoa.

        Itera sobre cada ``person_id`` presente nos registros e calcula
        as mesmas métricas de ``get_summary()`` filtradas para aquela pessoa.

        Parâmetros
        ----------
        elapsed_seconds : float
            Duração total da sessão em segundos.

        Retorna
        -------
        dict[int, dict]
            Dicionário mapeando cada ``person_id`` ao seu sumário individual.
            Retorna dicionário vazio se nenhum frame foi registrado.
        """
        if not self.records:
            return {}

        df   = self.to_dataframe()
        pids = sorted(df["person_id"].unique())
        result = {}

        for pid in pids:
            sub_df  = df[df["person_id"] == pid]
            sub_det = sub_df[sub_df["face_detected"]]

            blinks    = sub_df[sub_df["blink_occurred"]]
            total_b   = len(blinks)
            voluntary = int((blinks["blink_type"] == "voluntary").sum())
            blink_rate = total_b / elapsed_seconds * 60.0 if elapsed_seconds > 0 else 0.0

            on_screen_pct  = float(sub_det["on_screen"].mean() * 100)   if len(sub_det) > 0 else 0.0
            face_stability = float(len(sub_det) / len(sub_df) * 100)    if len(sub_df) > 0 else 0.0
            gaze_counts    = sub_det["gaze_direction"].value_counts().to_dict() if len(sub_det) > 0 else {}
            emotion_counts = sub_det["dominant_emotion"].value_counts().to_dict() if len(sub_det) > 0 else {}
            dominant_emo   = next(iter(emotion_counts), "unknown")
            sudden_moves   = int(sub_det["sudden_movement"].sum())       if len(sub_det) > 0 else 0
            avg_yaw        = float(sub_det["head_yaw"].abs().mean())     if len(sub_det) > 0 else 0.0
            avg_pitch      = float(sub_det["head_pitch"].abs().mean())   if len(sub_det) > 0 else 0.0

            breath_det    = sub_det[sub_det["respiratory_rate"] > 0]
            avg_resp_rate = float(breath_det["respiratory_rate"].mean()) if len(breath_det) > 0 else 0.0
            avg_breath_reg = float(sub_det["breath_regularity"].mean())  if len(sub_det) > 0 else 0.0

            fat_det       = sub_det[sub_det["fatigue_level"] != "unknown"]
            avg_fat       = float(fat_det["fatigue_score"].mean())       if len(fat_det) > 0 else 0.0
            dom_fat       = next(iter(fat_det["fatigue_level"].value_counts().index), "unknown") if len(fat_det) > 0 else "unknown"

            micros        = sub_det[sub_det["microexp_detected"]]
            total_micros  = len(micros)
            avg_mi_int    = float(micros["microexp_intensity"].mean())   if total_micros > 0 else 0.0

            asym_det      = sub_det[sub_det["asymmetry_score"] > 0]
            avg_asym      = float(asym_det["asymmetry_score"].mean())    if len(asym_det) > 0 else 0.0

            result[pid] = {
                "total_frames":           len(sub_df),
                "face_stability_pct":     face_stability,
                "total_blinks":           total_b,
                "voluntary_blinks":       voluntary,
                "involuntary_blinks":     total_b - voluntary,
                "blink_rate_per_min":     blink_rate,
                "on_screen_pct":          on_screen_pct,
                "gaze_distribution":      gaze_counts,
                "dominant_emotion":       dominant_emo,
                "emotion_distribution":   emotion_counts,
                "sudden_movements":       sudden_moves,
                "avg_head_yaw_deg":       avg_yaw,
                "avg_head_pitch_deg":     avg_pitch,
                "avg_respiratory_rate":   avg_resp_rate,
                "avg_breath_regularity":  avg_breath_reg,
                "avg_fatigue_score":      avg_fat,
                "dominant_fatigue":       dom_fat,
                "total_microexpressions": total_micros,
                "avg_micro_intensity":    avg_mi_int,
                "avg_asymmetry_score":    avg_asym,
            }

        return result
