"""
train.py — Entrenamiento y registro del modelo de clustering para Wholesale Customers.

Reproduce en código de producción la configuración ganadora decidida en
notebooks/03_modelado_experiment_tracking.ipynb (Secciones J y K del enunciado):
KMeans con k=3 sobre 5 componentes PCA.

Cada ejecución crea un NUEVO run en MLflow (no sobrescribe corridas anteriores),
para mantener el historial completo necesario para la estrategia de reentrenamiento
(Sección R) y el monitoreo de drift (Sección O/P).

Uso como script:
    python src/training/train.py

Uso como módulo (desde otro script del pipeline, ej. una API o un job de reentrenamiento):
    from src.training.train import entrenar_y_registrar
    resultado = entrenar_y_registrar()
"""

import json
import logging
import sys
import tempfile
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [training] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent and not (p / "src").exists():
        p = p.parent
    if not (p / "src").exists():
        raise RuntimeError("No se encontró el directorio raíz del repo (src/)")
    return p


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

from src.data_quality.validate import validar_calidad_datos
from src.features.build_features import COLS_GASTO, FeatureBuilder

# --------------------------------------------------------------------
# Configuración del modelo ganador (decidida y justificada en el notebook)
# --------------------------------------------------------------------
K_PCA = 5
N_CLUSTERS = 3
ALGORITHM = "KMeans"
RANDOM_SEED = 42
SEMILLA_VALIDACION = 123  # semilla alternativa para la prueba de estabilidad
UMBRAL_ESTABILIDAD = 0.05  # variación máxima aceptable de silhouette entre semillas

MLFLOW_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "wholesale-clustering-grupo6"
MODEL_NAME = "wholesale-clustering-grupo6"


def cargar_datos_validados() -> tuple[pd.DataFrame, str]:
    """Carga el dataset crudo, valida su calidad y devuelve (df, data_version)."""
    raw_dir = REPO_ROOT / "data" / "raw"

    df_raw = pd.read_csv(raw_dir / "wholesale_customers_raw.csv")
    with open(raw_dir / "ingestion_metadata.json", "r", encoding="utf-8") as f:
        data_version = json.load(f)["data_version"]

    reporte = validar_calidad_datos(df_raw, COLS_GASTO)
    for regla, resultado in reporte.items():
        estado = "PASS" if resultado["pass"] else "FAIL"
        logger.info("%s | %s: %s", estado, regla, resultado["detalle"])

    if not all(r["pass"] for r in reporte.values()):
        raise RuntimeError("Data Quality Gates falló — no se debe entrenar sobre estos datos.")

    return df_raw, data_version


def entrenar_y_registrar() -> dict:
    """Ejecuta el ciclo completo: datos -> features -> entrenamiento -> MLflow -> registry.

    Crea un nuevo run en cada ejecución. Devuelve un dict con run_id, model_version
    y si el modelo fue promovido a Production.
    """
    df_raw, data_version = cargar_datos_validados()

    fb = FeatureBuilder(n_pca_components=K_PCA)
    df_features = fb.fit_transform(df_raw)
    X = df_features[fb.cols_finales_]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"{ALGORITHM}_k{N_CLUSTERS}_train"):
        modelo = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
        labels = modelo.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)

        mlflow.log_params({
            "algorithm": ALGORITHM,
            "n_clusters": N_CLUSTERS,
            "feature_set": f"PCA_{K_PCA}_componentes",
            "random_seed": RANDOM_SEED,
            "data_version": data_version,
        })
        mlflow.log_metric("silhouette", silhouette)
        mlflow.log_metric("davies_bouldin", davies_bouldin)
        mlflow.log_metric("inertia", modelo.inertia_)
        mlflow.sklearn.log_model(modelo, "model")

        # FeatureBuilder también se loguea como artifact de este run (no solo
        # se exporta a models/ al promover), para que cualquier run del
        # historial —no solo el promovido a Production— sea reproducible
        # directamente desde MLflow, sin depender de la copia local en disco.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "feature_builder.joblib"
            fb.save(tmp_path)
            mlflow.log_artifact(str(tmp_path), artifact_path="feature_builder")

        run_id = mlflow.active_run().info.run_id
        logger.info(
            "Run registrado: %s | silhouette=%.3f | davies_bouldin=%.3f",
            run_id, silhouette, davies_bouldin,
        )

        # --- Validación de estabilidad ante cambio de semilla ---
        modelo_val = KMeans(n_clusters=N_CLUSTERS, random_state=SEMILLA_VALIDACION, n_init=10)
        labels_val = modelo_val.fit_predict(X)
        silhouette_val = silhouette_score(X, labels_val)
        diferencia = abs(silhouette_val - silhouette)
        paso_validacion = diferencia < UMBRAL_ESTABILIDAD

        mlflow.log_metric("silhouette_semilla_alternativa", silhouette_val)
        mlflow.log_metric("diferencia_estabilidad", diferencia)

        logger.info(
            "Validación de estabilidad: %.3f vs %.3f (diferencia %.3f, umbral %.2f) -> %s",
            silhouette, silhouette_val, diferencia, UMBRAL_ESTABILIDAD,
            "PASS" if paso_validacion else "FAIL",
        )

        # --- Model Registry: Experiment -> Candidate -> Validation -> Production ---
        model_version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, model_version.version, "lifecycle_stage", "candidate")

        promovido = False
        if paso_validacion:
            client.set_model_version_tag(MODEL_NAME, model_version.version, "lifecycle_stage", "validation")
            # El alias "production" es único por modelo: asignarlo aquí se lo quita
            # automáticamente a cualquier versión anterior que lo tuviera.
            client.set_registered_model_alias(MODEL_NAME, "production", model_version.version)
            client.set_model_version_tag(MODEL_NAME, model_version.version, "lifecycle_stage", "production")
            promovido = True
            logger.info("Modelo promovido a Production: %s v%s", MODEL_NAME, model_version.version)

            # --- Exportación servible (sección M) ---
            # MLflow gobierna el tracking y el registro de versiones, pero la API
            # de inferencia carga una copia liviana en joblib, para no depender de
            # acceso a la base de datos de MLflow dentro del contenedor Docker.
            models_dir = REPO_ROOT / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(modelo, models_dir / "kmeans_production.joblib")
            fb.save(models_dir / "feature_builder.joblib")

            production_metadata = {
                "model_name": MODEL_NAME,
                "model_version": model_version.version,
                "run_id": run_id,
                "algorithm": ALGORITHM,
                "n_clusters": N_CLUSTERS,
                "k_pca": K_PCA,
                "data_version": data_version,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
            }
            with open(models_dir / "production_metadata.json", "w", encoding="utf-8") as f:
                json.dump(production_metadata, f, indent=2, ensure_ascii=False)

            logger.info("Modelo exportado para serving en: %s", models_dir)
        else:
            logger.warning(
                "Modelo NO promovido a Production (falló validación de estabilidad): %s v%s",
                MODEL_NAME, model_version.version,
            )

    return {
        "run_id": run_id,
        "model_version": model_version.version,
        "promovido_a_produccion": promovido,
    }


if __name__ == "__main__":
    resultado = entrenar_y_registrar()
    logger.info("Entrenamiento finalizado: %s", resultado)