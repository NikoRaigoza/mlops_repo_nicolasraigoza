"""Valida la estructura mínima del repositorio exigida por el enunciado."""

import pytest

from conftest import RAIZ

ARCHIVOS_OBLIGATORIOS = [
    "mlops_pipeline/Cargar_datos.ipynb",
    "mlops_pipeline/comprension_eda.ipynb",
    "mlops_pipeline/ft_engineering.py",
    "mlops_pipeline/model_training_evaluation.py",
    "mlops_pipeline/model_deploy.py",
    "mlops_pipeline/model_monitoring.py",
    "Base_de_datos.csv",
    "requirements.txt",
    ".gitignore",
    "setup.bat",
    "readme.md",
]


@pytest.mark.parametrize("ruta", ARCHIVOS_OBLIGATORIOS)
def test_archivo_obligatorio_existe(ruta):
    assert (RAIZ / ruta).is_file(), f"Falta el archivo obligatorio: {ruta}"