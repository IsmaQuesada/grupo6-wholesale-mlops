"""
main.py — API de inferencia para el modelo de clustering de Wholesale Customers.

Carga el modelo K-Means y el FeatureBuilder exportados por
src/training/train.py (sección M del enunciado) y expone un endpoint
/predict que recibe el gasto de un cliente nuevo y devuelve el cluster
asignado, siguiendo el formato de respuesta especificado para clustering:

    { "cluster": 4, "distance_to_centroid": 0.38, "model_version": "5" }

Uso local:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Documentación interactiva (generada automáticamente por FastAPI):
    http://localhost:8000/docs
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import ClienteInput, HealthOutput, PrediccionOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# --------------------------------------------------------------------
# Carga de artefactos al iniciar la aplicación (una sola vez, no por request)
# --------------------------------------------------------------------
_modelo = None
_feature_builder = None
_metadata = None


def cargar_artefactos():
    global _modelo, _feature_builder, _metadata

    modelo_path = MODELS_DIR / "kmeans_production.joblib"
    builder_path = MODELS_DIR / "feature_builder.joblib"
    metadata_path = MODELS_DIR / "production_metadata.json"

    if not modelo_path.exists() or not builder_path.exists():
        logger.error(
            "No se encontraron los artefactos del modelo en %s. "
            "Corre primero: python src/training/train.py",
            MODELS_DIR,
        )
        raise RuntimeError(
            f"Modelo no encontrado en {MODELS_DIR}. "
            "Ejecuta src/training/train.py antes de levantar la API."
        )

    _modelo = joblib.load(modelo_path)
    _feature_builder = joblib.load(builder_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    logger.info(
        "Modelo cargado: %s v%s (silhouette=%.3f)",
        _metadata["model_name"], _metadata["model_version"], _metadata["silhouette"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_artefactos()
    yield


app = FastAPI(
    title="Wholesale Customers — Clustering API",
    description="API de inferencia para segmentación de clientes mayoristas (Grupo 6)",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------
@app.get("/health", response_model=HealthOutput)
def health():
    """Verifica que la API esté viva y el modelo cargado correctamente."""
    return HealthOutput(
        status="ok",
        model_version=str(_metadata["model_version"]) if _metadata else "unknown",
        model_loaded=_modelo is not None,
    )


@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):
    """
    Recibe el gasto anual de un cliente en las 6 categorías y devuelve el
    cluster asignado junto con la distancia a su centroide.
    """
    if _modelo is None or _feature_builder is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Intente más tarde.")

    try:
        df_input = pd.DataFrame([cliente.model_dump()])
        # FeatureBuilder conserva Channel/Region en su salida para permitir
        # el perfilado de clusters (ver notebooks 02/03), aunque NO forman
        # parte de las columnas que entrenan o predicen el modelo. Un
        # cliente nuevo llega sin esos datos, así que se completan como
        # nulos únicamente para satisfacer la forma esperada por
        # FeatureBuilder — no afectan el cluster ni la distancia calculada.
        df_input["Channel"] = np.nan
        df_input["Region"] = np.nan

        df_features = _feature_builder.transform(df_input)
        X = df_features[_feature_builder.cols_finales_]

        cluster = int(_modelo.predict(X)[0])
        distancias = _modelo.transform(X)[0]  # distancia a CADA centroide
        distancia_al_asignado = float(distancias[cluster])

    except Exception as exc:  # noqa: BLE001
        logger.error("Error al procesar la predicción: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo procesar el input recibido: {exc}",
        ) from exc

    return PrediccionOutput(
        cluster=cluster,
        distance_to_centroid=round(distancia_al_asignado, 4),
        model_version=str(_metadata["model_version"]),
    )


@app.get("/")
def root():
    return {
        "message": "Wholesale Customers Clustering API — Grupo 6",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
