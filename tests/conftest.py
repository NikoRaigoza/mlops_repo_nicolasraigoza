"""Configuración común de las pruebas."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_PIPELINE = RAIZ / "mlops_pipeline"
sys.path.insert(0, str(CARPETA_PIPELINE))