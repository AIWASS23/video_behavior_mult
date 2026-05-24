"""
Estimativa de frequência respiratória a partir do movimento das narinas.

Fundamento científico
---------------------
Durante a inspiração, os músculos nasalares alargam levemente as narinas
(dilatação alar). Na expiração, elas retornam ao estado neutro. Esse movimento
sutil (~0,5–3 % da largura facial) é detectável como uma oscilação periódica
na distância entre os pontos alares externos do nariz.

A frequência respiratória normal em adultos em repouso é de 12–20 ciclos/minuto
(0,20–0,33 Hz). O pipeline aplica:
  1. Cálculo da largura nasal normalizada (landmarks 129 e 358)
  2. Suavização por média móvel ponderada para remover ruído de landmarks
  3. Filtro passa-banda Butterworth (0,10–0,60 Hz) sobre o buffer acumulado
  4. Detecção de picos na série filtrada (cada pico = um ciclo respiratório)
  5. Derivação de taxa, profundidade e regularidade

Limitações
----------
- O sinal é sutil: resolução de câmera, iluminação e compressão de vídeo
  afetam a qualidade de detecção.
- Requer pelo menos 10 s de dados para estimativa estável.
- Máscara, barba densa ou oclusão nasal prejudicam a detecção.
- A profundidade é relativa ao histórico da própria sessão, não calibrada.
- Não deve ser usado para fins médicos.

Referências
-----------
- Procházka et al., "Nostril Movement as a Respiratory Indicator in Facial
  Video", IEEE EMBC 2021.
- Al-Khalidi et al., "Respiration Rate Monitoring Methods: A Review",
  Pediatric Pulmonology, 2011.

Índices dos landmarks MediaPipe utilizados
------------------------------------------
  129 → base alar esquerda externa (ponto mais largo da narina esquerda)
  358 → base alar direita externa  (ponto mais largo da narina direita)
   33 → canto externo do olho direito  ┐ usados para
  263 → canto externo do olho esquerdo ┘ normalização da largura facial
"""


import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .face_tracker import FaceData

log = logging.getLogger(__name__)

# ── Índices de landmarks ──────────────────────────────────────────────────────
_ALAR_LEFT   = 129   # base alar esquerda externa
_ALAR_RIGHT  = 358   # base alar direita externa
_EYE_RIGHT   =  33   # canto externo olho direito  (referência de largura facial)
_EYE_LEFT    = 263   # canto externo olho esquerdo (referência de largura facial)

# Frequências de interesse para respiração (Hz)
_FREQ_LOW  = 0.10   # ~6 resp/min  (mínimo fisiológico para adulto em repouso)
_FREQ_HIGH = 0.60   # ~36 resp/min (máximo para respiração acelerada)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BreathingResult:
    """
    Métricas respiratórias estimadas para o frame atual.

    Atributos
    ----------
    respiratory_rate : float
        Taxa respiratória estimada em ciclos por minuto.
        0.0 enquanto não há dados suficientes (< 10 s de buffer).
    breath_depth : str
        Profundidade relativa da respiração: ``'shallow'`` (superficial),
        ``'normal'`` ou ``'deep'`` (profunda), baseada na amplitude do sinal.
    breath_phase : str
        Fase atual estimada: ``'inhale'`` (inspiração) quando as narinas
        estão se alargando, ``'exhale'`` (expiração) quando se estreitando,
        ``'hold'`` quando o sinal está estável.
    signal_amplitude : float
        Amplitude normalizada do sinal de narina no buffer atual.
        Valor em [0, 1]; quanto maior, mais visível o movimento respiratório.
    regularity : float
        Regularidade dos intervalos entre picos em [0, 1].
        1.0 = respiração perfeitamente regular; 0.0 = muito irregular.
    is_reliable : bool
        True quando há dados suficientes e o sinal tem qualidade mínima
        para produzir uma estimativa confiável.
    nostril_width_norm : float
        Largura nasal normalizada pelo espaçamento inter-ocular no frame atual.
        Útil para visualização do sinal bruto.
    """

    respiratory_rate: float
    breath_depth: str
    breath_phase: str
    signal_amplitude: float
    regularity: float
    is_reliable: bool
    nostril_width_norm: float


class BreathingDetector:
    """
    Detector de respiração baseado em variação da largura das narinas.

    Acumula um buffer deslizante do sinal de largura nasal normalizada,
    aplica filtragem passa-banda e detecta picos para estimar a frequência
    respiratória e outras métricas.

    Parâmetros
    ----------
    buffer_seconds : float
        Duração máxima do buffer de sinal em segundos (padrão: 20).
        Buffers maiores permitem estimativas mais robustas mas aumentam
        a latência para a primeira leitura.
    min_seconds_for_rate : float
        Tempo mínimo de buffer antes de calcular a taxa respiratória
        (padrão: 10). Abaixo disso, ``is_reliable=False``.
    smoothing_window : int
        Tamanho da janela de média móvel para suavização do sinal bruto
        em número de amostras (padrão: 5). Reduz ruído de tracking de
        landmarks sem distorcer muito a frequência respiratória.
    """

    def __init__(
        self,
        buffer_seconds: float = 20.0,
        min_seconds_for_rate: float = 10.0,
        smoothing_window: int = 5,
    ):
        """
        Inicializa o detector e reserva os buffers de sinal.

        Parâmetros
        ----------
        buffer_seconds : float, opcional
            Duração máxima do buffer deslizante (padrão: 20.0 s).
        min_seconds_for_rate : float, opcional
            Segundos mínimos de dados para produzir estimativa confiável
            (padrão: 10.0 s).
        smoothing_window : int, opcional
            Janela de suavização em amostras (padrão: 5).
        """
        self.buffer_seconds       = buffer_seconds
        self.min_seconds_for_rate = min_seconds_for_rate
        self.smoothing_window     = smoothing_window

        # Buffer deslizante: usa maxlen dinâmico baseado em fps observado
        self._times:   deque[float] = deque()
        self._widths:  deque[float] = deque()  # largura nasal bruta normalizada
        self._smooth:  deque[float] = deque()  # sinal suavizado

        self._breath_intervals: deque[float] = deque(maxlen=15)
        self._last_peak_time: Optional[float] = None
        self.current: Optional[BreathingResult] = None

    # ── API pública ───────────────────────────────────────────────────────────

    def update(self, face_data: FaceData, timestamp: float) -> Optional[BreathingResult]:
        """
        Atualiza o buffer com os dados do frame atual e retorna as métricas.

        Deve ser chamado em ordem cronológica a cada frame com rosto detectado.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais do frame atual. Requer ``detected=True``;
            retorna None caso contrário.
        timestamp : float
            Tempo em segundos desde o início da sessão.

        Retorna
        -------
        BreathingResult ou None
            Métricas respiratórias calculadas, ou None se nenhum rosto
            foi detectado.
        """
        if not face_data.detected:
            return self.current

        width_norm = self._compute_nostril_width(face_data)
        if width_norm is None:
            return self.current

        # Adiciona ao buffer e remove amostras antigas
        self._times.append(timestamp)
        self._widths.append(width_norm)
        self._trim_buffer()

        # Suavização por média móvel
        window = min(self.smoothing_window, len(self._widths))
        smooth_val = float(np.mean(list(self._widths)[-window:]))
        self._smooth.append(smooth_val)

        # Calcula as métricas
        result = self._compute_metrics(timestamp, width_norm, smooth_val)
        self.current = result
        return result

    def get_stats(self) -> dict:
        """
        Retorna estatísticas agregadas da sessão respiratória.

        Retorna
        -------
        dict
            Dicionário com as chaves:
            - ``avg_respiratory_rate``   : float — taxa média em ciclos/min
            - ``min_respiratory_rate``   : float — taxa mínima registrada
            - ``max_respiratory_rate``   : float — taxa máxima registrada
            - ``avg_interval_s``         : float — intervalo médio entre ciclos (s)
            - ``breathing_regularity``   : float — regularidade média [0, 1]
            - ``total_breaths_detected`` : int   — total de ciclos detectados
        """
        if not self._breath_intervals:
            return {
                "avg_respiratory_rate":   0.0,
                "min_respiratory_rate":   0.0,
                "max_respiratory_rate":   0.0,
                "avg_interval_s":         0.0,
                "breathing_regularity":   0.0,
                "total_breaths_detected": 0,
            }
        intervals = list(self._breath_intervals)
        rates = [60.0 / i for i in intervals if i > 0]
        reg   = self._regularity(intervals)
        return {
            "avg_respiratory_rate":   float(np.mean(rates))   if rates else 0.0,
            "min_respiratory_rate":   float(np.min(rates))    if rates else 0.0,
            "max_respiratory_rate":   float(np.max(rates))    if rates else 0.0,
            "avg_interval_s":         float(np.mean(intervals)),
            "breathing_regularity":   reg,
            "total_breaths_detected": len(self._breath_intervals),
        }

    def get_signal_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna os arrays de tempo e sinal suavizado acumulados no buffer.

        Útil para gerar gráficos do sinal respiratório bruto.

        Retorna
        -------
        tuple[np.ndarray, np.ndarray]
            Par (timestamps, sinal_suavizado) como arrays NumPy 1-D.
        """
        return np.array(self._times), np.array(self._smooth)

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _compute_nostril_width(self, face_data: FaceData) -> Optional[float]:
        """
        Calcula a largura nasal normalizada pelo espaçamento inter-ocular.

        A largura nasal bruta (distância entre as bases alares externas,
        landmarks 129 e 358) é dividida pela distância entre os cantos
        externos dos olhos (landmarks 33 e 263) para compensar variações
        de distância da câmera e movimentos de cabeça.

        Parâmetros
        ----------
        face_data : FaceData
            Dados faciais com landmarks do MediaPipe.

        Retorna
        -------
        float ou None
            Largura nasal normalizada em [0, ~0.6], ou None se os landmarks
            não estiverem disponíveis.
        """
        try:
            lm = face_data.landmarks
            alar_l = np.array([lm[_ALAR_LEFT].x,  lm[_ALAR_LEFT].y])
            alar_r = np.array([lm[_ALAR_RIGHT].x, lm[_ALAR_RIGHT].y])
            eye_l  = np.array([lm[_EYE_LEFT].x,   lm[_EYE_LEFT].y])
            eye_r  = np.array([lm[_EYE_RIGHT].x,  lm[_EYE_RIGHT].y])

            nostril_w = float(np.linalg.norm(alar_l - alar_r))
            inter_ocu = float(np.linalg.norm(eye_l  - eye_r))

            if inter_ocu < 1e-6:
                return None
            return nostril_w / inter_ocu
        except Exception:
            return None

    def _trim_buffer(self):
        """
        Remove amostras mais antigas que ``buffer_seconds`` do buffer deslizante.

        Mantém o buffer dentro do tamanho configurado para evitar crescimento
        ilimitado de memória durante sessões longas.
        """
        if len(self._times) < 2:
            return
        cutoff = self._times[-1] - self.buffer_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._widths.popleft()
            if self._smooth:
                self._smooth.popleft()

    def _compute_metrics(
        self, timestamp: float, raw_width: float, smooth_val: float
    ) -> BreathingResult:
        """
        Calcula todas as métricas respiratórias a partir do buffer atual.

        Aplica filtro passa-banda Butterworth (se scipy disponível) ou
        análise simples de amplitude sobre o buffer suavizado.
        Detecta picos no sinal filtrado e deriva taxa e regularidade.

        Parâmetros
        ----------
        timestamp : float
            Timestamp do frame atual em segundos.
        raw_width : float
            Largura nasal normalizada atual (sinal bruto).
        smooth_val : float
            Valor suavizado atual (média móvel do buffer recente).

        Retorna
        -------
        BreathingResult
            Métricas calculadas para o frame atual.
        """
        n = len(self._times)
        elapsed = self._times[-1] - self._times[0] if n > 1 else 0.0

        # Sinal suavizado como array
        sig = np.array(self._smooth)
        amplitude = float(sig.max() - sig.min()) if len(sig) > 1 else 0.0

        is_reliable = elapsed >= self.min_seconds_for_rate and amplitude > 1e-4

        # ── Taxa respiratória ────────────────────────────────────────────────
        resp_rate = 0.0
        if is_reliable and n >= 4:
            resp_rate = self._estimate_rate(sig, self._times)

        # ── Regularidade ─────────────────────────────────────────────────────
        regularity = self._regularity(list(self._breath_intervals))

        # ── Fase (inspiração / expiração) ─────────────────────────────────────
        phase = self._detect_phase(sig)

        # ── Profundidade ──────────────────────────────────────────────────────
        depth = self._classify_depth(amplitude)

        return BreathingResult(
            respiratory_rate=resp_rate,
            breath_depth=depth,
            breath_phase=phase,
            signal_amplitude=amplitude,
            regularity=regularity,
            is_reliable=is_reliable,
            nostril_width_norm=raw_width,
        )

    def _estimate_rate(
        self, sig: np.ndarray, times: deque
    ) -> float:
        """
        Estima a taxa respiratória por detecção de picos no sinal filtrado.

        Tenta aplicar filtro Butterworth passa-banda via scipy; se não
        disponível, usa o sinal suavizado diretamente. Os picos detectados
        representam momentos de máxima dilatação nasal (pico de inspiração).

        Parâmetros
        ----------
        sig : np.ndarray
            Array 1-D com o sinal suavizado do buffer atual.
        times : deque
            Buffer de timestamps correspondentes ao sinal.

        Retorna
        -------
        float
            Taxa respiratória estimada em ciclos por minuto, ou 0.0 se
            não for possível detectar picos suficientes.
        """
        t_arr = np.array(times)
        elapsed = t_arr[-1] - t_arr[0]
        if elapsed < 1.0:
            return 0.0

        fps_est = len(sig) / elapsed

        # Filtro passa-banda Butterworth (tenta scipy)
        filtered = sig.copy()
        try:
            from scipy.signal import butter, filtfilt
            nyq  = fps_est / 2.0
            low  = _FREQ_LOW  / nyq
            high = min(_FREQ_HIGH / nyq, 0.95)
            if 0 < low < high < 1:
                b, a     = butter(2, [low, high], btype="band")
                filtered = filtfilt(b, a, sig, padlen=min(len(sig) // 3, 10))
        except Exception:
            pass  # usa sinal suavizado sem filtragem adicional

        # Detecção de picos com distância mínima de 1.5 s
        try:
            from scipy.signal import find_peaks
            min_dist = max(1, int(fps_est * 1.5))
            peaks, _ = find_peaks(filtered, distance=min_dist,
                                  prominence=filtered.std() * 0.3)
        except Exception:
            # fallback: cruza média ascendente
            mean = filtered.mean()
            peaks = np.where(
                (filtered[1:-1] > mean) &
                (filtered[1:-1] > filtered[:-2]) &
                (filtered[1:-1] > filtered[2:])
            )[0] + 1

        if len(peaks) < 2:
            return 0.0

        # Atualiza intervalos entre picos para regularidade
        peak_times = t_arr[peaks]
        new_intervals = np.diff(peak_times)
        for iv in new_intervals:
            if 1.0 < iv < 10.0:  # intervalo fisiologicamente plausível
                self._breath_intervals.append(float(iv))
                self._last_peak_time = float(peak_times[-1])

        # Taxa = número de picos / tempo decorrido * 60
        rate = len(peaks) / elapsed * 60.0
        # Restringe ao intervalo fisiológico
        return float(np.clip(rate, 4.0, 40.0))

    def _detect_phase(self, sig: np.ndarray) -> str:
        """
        Estima a fase respiratória atual (inspiração, expiração ou pausa).

        Compara os últimos valores do sinal com a média recente para
        determinar se as narinas estão se alargando (inspiração) ou
        se estreitando (expiração).

        Parâmetros
        ----------
        sig : np.ndarray
            Sinal suavizado do buffer atual.

        Retorna
        -------
        str
            ``'inhale'``, ``'exhale'`` ou ``'hold'``.
        """
        if len(sig) < 6:
            return "hold"
        recent  = sig[-3:]
        earlier = sig[-6:-3]
        delta   = float(recent.mean() - earlier.mean())
        thresh  = sig.std() * 0.15
        if delta > thresh:
            return "inhale"
        if delta < -thresh:
            return "exhale"
        return "hold"

    @staticmethod
    def _classify_depth(amplitude: float) -> str:
        """
        Classifica a profundidade respiratória com base na amplitude do sinal.

        Os limiares são empíricos e relativos ao sinal normalizado por largura
        facial. Respirações mais profundas produzem maior dilatação alar
        e, portanto, maior amplitude no sinal.

        Parâmetros
        ----------
        amplitude : float
            Amplitude pico-a-pico do sinal suavizado no buffer.

        Retorna
        -------
        str
            ``'shallow'`` (< 0.008), ``'normal'`` (0.008–0.020) ou
            ``'deep'`` (≥ 0.020).
        """
        if amplitude < 0.008:
            return "shallow"
        if amplitude < 0.020:
            return "normal"
        return "deep"

    @staticmethod
    def _regularity(intervals: list[float]) -> float:
        """
        Calcula a regularidade respiratória como coeficiente de variação invertido.

        Um coeficiente de variação (CV = desvio_padrão / média) baixo indica
        respiração regular. O resultado é mapeado para [0, 1] onde 1 = regular.

        Parâmetros
        ----------
        intervals : list[float]
            Lista de intervalos em segundos entre ciclos respiratórios consecutivos.

        Retorna
        -------
        float
            Valor de regularidade em [0, 1]. Retorna 0.5 quando há dados
            insuficientes (menos de 3 intervalos).
        """
        if len(intervals) < 3:
            return 0.5
        arr = np.array(intervals)
        cv  = arr.std() / arr.mean() if arr.mean() > 0 else 1.0
        return float(np.clip(1.0 - cv, 0.0, 1.0))
