"""
Baixa os arquivos de modelo necessários para o pipeline de análise visual.

Modelos baixados
----------------
- face_landmarker.task  : MediaPipe Face Landmarker (478 landmarks + íris)
  Fonte: Google MediaPipe Model Cards
  Tamanho aproximado: 6 MB

Uso
---
  python download_models.py
"""

import os
import urllib.request

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def download(name: str, url: str):
    """Baixa um arquivo de modelo para o diretório models/, com barra de progresso."""
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest):
        print(f"  [ok] {name} já existe — ignorando download.")
        return

    print(f"  Baixando {name} ...")

    def _progress(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100)
        print(f"\r    {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  [ok] {name} salvo em {dest}")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Baixando modelos para o diretório models/ ...")
    for model_name, model_url in MODELS.items():
        download(model_name, model_url)
    print("\nPronto. Execute  python main.py  para iniciar a análise.")
