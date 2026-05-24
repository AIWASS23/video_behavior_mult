"""
Ponto de entrada principal do sistema de Análise de Comportamento Visual
com suporte a múltiplas pessoas simultâneas.

Orquestra o pipeline completo: leitura do vídeo (webcam ou arquivo),
rastreamento de identidade por centróide (PersonTracker), processamento
frame a frame de cada pessoa detectada, exibição do HUD ao vivo com painel
por pessoa e persistência dos resultados (CSV, vídeo anotado, relatório PDF).

Exemplos de uso
---------------
  # Webcam padrão (índice 0)
  python main.py

  # Arquivo de vídeo pré-gravado
  python main.py --source caminho/para/video.mp4

  # Webcam, sem janela de exibição (execução em segundo plano)
  python main.py --no-display

  # Captura por 30 segundos e gera relatório automaticamente
  python main.py --no-display --duration 30

Pressione  q  na janela de visualização ou  Ctrl-C  no terminal para encerrar.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import cv2
import yaml

# Garante que o pacote src seja importável ao executar da raiz do projeto
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from src.face_tracker import FaceTracker, FaceData
from src.person_tracker import PersonTracker
from src.person_manager import PersonManager
from src.metrics_collector import MetricsCollector
from src.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Funções auxiliares ────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    """
    Carrega o arquivo de configuração YAML e retorna como dicionário.

    Parâmetros
    ----------
    path : str
        Caminho para o arquivo YAML de configuração.

    Retorna
    -------
    dict
        Dicionário com todos os parâmetros de configuração.

    Levanta
    -------
    FileNotFoundError
        Se o arquivo não existir no caminho especificado.
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


def _crop_face(frame_bgr: np.ndarray, face_data: FaceData, padding: float = 0.30) -> np.ndarray:
    """
    Recorta a região do rosto no frame com margem de padding.

    Calcula o bounding box a partir de todos os 478 landmarks normalizados,
    expande com a margem indicada e retorna o trecho BGR recortado.

    Parâmetros
    ----------
    frame_bgr : np.ndarray
        Frame completo em BGR.
    face_data : FaceData
        Dados da detecção facial com landmarks normalizados.
    padding : float
        Proporção da margem extra ao redor do bounding box (padrão: 0.30).

    Retorna
    -------
    np.ndarray
        Imagem BGR recortada ao redor do rosto.
    """
    h, w = frame_bgr.shape[:2]
    xs = [lm.x for lm in face_data.landmarks]
    ys = [lm.y for lm in face_data.landmarks]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    x1 = max(0, int((min(xs) - padding * bw) * w))
    x2 = min(w, int((max(xs) + padding * bw) * w))
    y1 = max(0, int((min(ys) - padding * bh) * h))
    y2 = min(h, int((max(ys) + padding * bh) * h))
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame_bgr


def _build_video_writer(
    output_dir: str, w: int, h: int, fps: float
) -> tuple:
    """
    Cria e configura um VideoWriter do OpenCV para salvar o vídeo anotado.

    Parâmetros
    ----------
    output_dir : str
        Diretório de destino (deve existir previamente).
    w : int
        Largura do vídeo de saída em pixels.
    h : int
        Altura do vídeo de saída em pixels.
    fps : float
        Taxa de quadros do vídeo de saída.

    Retorna
    -------
    tuple[cv2.VideoWriter, str]
        Instância do VideoWriter e caminho completo do arquivo criado.
    """
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    path   = os.path.join(output_dir, f"analysis_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    log.info("Vídeo de saída → %s", path)
    return writer, path


# ── Loop principal de processamento ──────────────────────────────────────────

def run(args: argparse.Namespace):
    """
    Executa o pipeline multi-pessoa de análise de comportamento visual.

    Inicializa o FaceTracker (MediaPipe, até 10 rostos), o PersonTracker
    (atribuição de IDs persistentes por centróide) e o PersonManager
    (instâncias de detectores independentes por pessoa). A cada frame:

    1. Captura e conversão RGB
    2. Detecção de todos os rostos (FaceTracker → list[FaceData])
    3. Atribuição de IDs persistentes (PersonTracker)
    4. Para cada pessoa: executa todos os detectores via PersonManager
    5. Registra métricas por pessoa (MetricsCollector)
    6. Renderiza HUD com painel por pessoa (Visualizer)
    7. Escreve frame anotado no vídeo de saída

    Ao final: salva CSV, gera relatório PDF com página de comparação entre
    pessoas e exibe sumário por pessoa no terminal.

    Parâmetros
    ----------
    args : argparse.Namespace
        Argumentos parsed da linha de comando.
    """
    config = _load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    source: int | str = int(args.source) if args.source.isdigit() else args.source

    # ── Abertura da fonte de vídeo ────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Não foi possível abrir a fonte '%s'", source)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Fonte: %s  |  %dx%d @ %.1f fps", source, w, h, fps_src)

    # ── Instanciação dos módulos do pipeline ──────────────────────────────
    tracker        = FaceTracker(max_faces=10)
    person_tracker = PersonTracker(max_distance=0.20, max_missing_frames=30)
    person_manager = PersonManager(config=config, fps=fps_src)
    collector      = MetricsCollector()
    vis            = Visualizer()

    writer = None
    if not args.no_save:
        writer, _ = _build_video_writer(args.output, w, h, fps_src)

    duration = getattr(args, "duration", None)
    if duration:
        log.info("Análise iniciada.  Duração: %ds.", duration)
    else:
        log.info("Análise iniciada.  Pressione  q  para encerrar.")

    frame_idx        = 0
    start_time       = time.perf_counter()
    max_persons_seen = 0
    first_frames: dict[int, np.ndarray] = {}   # person_id → crop do primeiro frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("Fim do fluxo de vídeo.")
                break

            timestamp = time.perf_counter() - start_time

            if duration and timestamp >= duration:
                log.info("Duração atingida (%.0f s).", duration)
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Detecção e rastreamento de todas as pessoas ───────────────
            faces_raw = tracker.process_frame(frame_rgb)       # list[FaceData]
            tracked   = person_tracker.update(faces_raw)       # list[(id, FaceData)]
            person_manager.cleanup(person_tracker.active_ids)

            max_persons_seen = max(max_persons_seen, len(tracked))

            persons_for_hud: list[tuple[int, dict]] = []

            for person_id, face_data in tracked:
                # Salva crop do primeiro frame em que a pessoa aparece
                if person_id not in first_frames:
                    first_frames[person_id] = _crop_face(frame, face_data)

                dets = person_manager.get(person_id)

                blink_event      = dets["blinker"].update(face_data, timestamp)
                gaze_result      = dets["gazer"].update(face_data)
                head_pose        = dets["poser"].update(face_data)
                emotion_result   = dets["emoter"].update(frame, face_data, frame_idx)
                breathing_result = dets["breather"].update(face_data, timestamp)
                fatigue_result   = dets["fatiguer"].update(face_data, dets["blinker"].current_ear, timestamp)
                micro_result     = dets["microer"].update(face_data, timestamp)
                asym_result      = dets["asymer"].update(face_data)

                # ── Registro de métricas ──────────────────────────────────
                collector.update(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    person_id=person_id,
                    face_data=face_data,
                    blink_event=blink_event,
                    blink_ear=dets["blinker"].current_ear,
                    gaze_result=gaze_result,
                    head_pose=head_pose,
                    emotion_result=emotion_result,
                    breathing_result=breathing_result,
                    fatigue_result=fatigue_result,
                    micro_result=micro_result,
                    asym_result=asym_result,
                )

                persons_for_hud.append((person_id, {
                    "face_data":        face_data,
                    "blink_detector":   dets["blinker"],
                    "gaze_result":      gaze_result,
                    "head_pose":        head_pose,
                    "emotion_result":   emotion_result,
                    "breathing_result": breathing_result,
                    "fatigue_result":   fatigue_result,
                    "micro_result":     micro_result,
                    "asym_result":      asym_result,
                }))

            # ── Renderização do HUD ───────────────────────────────────────
            annotated = vis.draw_overlay(
                frame.copy(),
                persons=persons_for_hud,
                elapsed=timestamp,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Visual Behavior Analysis Multi  [q = encerrar]", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Encerrado pelo usuário.")
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        log.info("Interrompido pelo teclado.")

    finally:
        elapsed = time.perf_counter() - start_time

        # ── Persistência dos resultados ───────────────────────────────────
        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.output, f"metrics_{ts_str}.csv")
        collector.save_csv(csv_path)

        summary    = collector.get_summary(elapsed)
        per_person = collector.get_summary_per_person(elapsed)
        vis.generate_report(collector, summary, args.output, elapsed,
                            first_frames=first_frames)

        # ── Sumário no terminal ───────────────────────────────────────────
        person_ids   = summary.get("person_ids", [])
        person_count = summary.get("person_count", 0)

        print("\n" + "=" * 60)
        print(f"  RESUMO DA SESSÃO  —  {person_count} pessoa(s) rastreada(s)")
        print("=" * 60)

        skip_keys = {"gaze_distribution", "emotion_distribution", "person_ids",
                     "breath_depth_dist", "breath_phase_dist", "fatigue_level_dist",
                     "microexpression_dist", "dominant_side_dist"}

        for k, v in summary.items():
            if k in skip_keys:
                continue
            label = k.replace("_", " ").capitalize()
            if isinstance(v, float):
                print(f"  {label:<36s}: {v:.2f}")
            else:
                print(f"  {label:<36s}: {v}")

        if person_count > 1:
            print("\n" + "─" * 60)
            print("  MÉTRICAS POR PESSOA")
            print("─" * 60)
            compare_keys = [
                ("total_blinks",        "Piscadas"),
                ("blink_rate_per_min",  "Taxa piscadas/min"),
                ("on_screen_pct",       "Atenção tela (%)"),
                ("dominant_emotion",    "Emoção dominante"),
                ("avg_respiratory_rate","Taxa resp. (/min)"),
                ("avg_fatigue_score",   "Score fadiga"),
                ("total_microexpressions", "Microexpressões"),
                ("avg_asymmetry_score", "Assimetria"),
            ]
            header = f"  {'Métrica':<28s}" + "".join(f"  P{pid:<6}" for pid in person_ids)
            print(header)
            print("  " + "-" * (28 + 8 * len(person_ids)))
            for key, label in compare_keys:
                row = f"  {label:<28s}"
                for pid in person_ids:
                    val = per_person.get(pid, {}).get(key, "N/A")
                    if isinstance(val, float):
                        row += f"  {val:<6.2f}"
                    else:
                        row += f"  {str(val):<6}"
                print(row)

        gd = summary.get("gaze_distribution", {})
        if gd:
            print("\n  Direções do olhar (global):")
            for d, cnt in sorted(gd.items(), key=lambda x: -x[1]):
                print(f"    {d:<18s}: {cnt}")
        print("=" * 60 + "\n")

        # ── Liberação de recursos ─────────────────────────────────────────
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        tracker.close()


# ── Interface de linha de comando ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Define e processa os argumentos da linha de comando.

    Retorna
    -------
    argparse.Namespace
        Objeto com os atributos: source, output, config, no_display,
        no_save, duration.
    """
    p = argparse.ArgumentParser(
        description="Análise de Comportamento Visual Multi-Pessoa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python main.py\n"
            "  python main.py --source video.mp4\n"
            "  python main.py --no-display --duration 30\n"
        ),
    )
    p.add_argument(
        "--source", default="0",
        help="Fonte de vídeo: 0 (padrão) para webcam, ou caminho para arquivo.",
    )
    p.add_argument(
        "--output", default="outputs",
        help="Diretório para salvar CSV, relatório PDF e vídeo anotado.",
    )
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Desativa a janela OpenCV ao vivo.",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Não salva o vídeo anotado (CSV e PDF ainda são gerados).",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Encerra automaticamente após N segundos.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
