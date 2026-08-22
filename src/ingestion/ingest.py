"""
ingest.py — Ingesta reproducible del dataset Wholesale Customers (UCI ID 292)

Uso:
    python src/ingestion/ingest.py

Estrategia de ingesta (en orden de preferencia):
    1. Paquete oficial `ucimlrepo` (mantenido por UCI, incluye metadata).
    2. Descarga directa del CSV crudo publicado en el repositorio UCI.
    3. Copia local de respaldo (`data/raw/_fallback/`) SOLO si no hay internet
       disponible durante una demo en vivo. Esta copia debe existir en el
       repo únicamente como último recurso documentado, nunca como fuente
       primaria de verdad.

El script es idempotente: puede ejecutarse múltiples veces y siempre
producirá el mismo archivo de salida versionado por fecha de ingesta.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FALLBACK_DIR = RAW_DIR / "_fallback"
OUTPUT_PATH = RAW_DIR / "wholesale_customers_raw.csv"
METADATA_PATH = RAW_DIR / "ingestion_metadata.json"

UCI_DATASET_ID = 292
UCI_DIRECT_CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00292/Wholesale%20customers%20data.csv"
)

EXPECTED_COLUMNS = [
    "Channel",
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]
EXPECTED_MIN_ROWS = 400  # el dataset original tiene 440 filas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingest] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Métodos de ingesta
# --------------------------------------------------------------------------
def fetch_via_ucimlrepo() -> pd.DataFrame:
    """Método primario: paquete oficial ucimlrepo."""
    from ucimlrepo import fetch_ucirepo

    logger.info("Intentando ingesta vía ucimlrepo (dataset id=%s)...", UCI_DATASET_ID)
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = dataset.data.features.copy()
    # ucimlrepo separa a veces features/targets; para este dataset todo
    # viene en features, pero validamos por robustez.
    if dataset.data.targets is not None and not dataset.data.targets.empty:
        df = pd.concat([df, dataset.data.targets], axis=1)
    logger.info("Ingesta vía ucimlrepo exitosa: %s filas.", len(df))
    return df


def fetch_via_direct_url() -> pd.DataFrame:
    """Método secundario: descarga directa del CSV crudo de UCI."""
    logger.info("Intentando ingesta vía URL directa: %s", UCI_DIRECT_CSV_URL)
    df = pd.read_csv(UCI_DIRECT_CSV_URL)
    logger.info("Ingesta vía URL directa exitosa: %s filas.", len(df))
    return df


def fetch_via_local_fallback() -> pd.DataFrame:
    """
    Último recurso: copia local versionada en data/raw/_fallback/.

    Esta copia debe subirse UNA sola vez al repo (es pequeña, ~440 filas)
    y documentarse en el README como plan de contingencia para demo sin
    internet. No sustituye la ingesta reproducible real.
    """
    fallback_file = FALLBACK_DIR / "wholesale_customers_data.csv"
    if not fallback_file.exists():
        raise FileNotFoundError(
            f"No hay conexión a internet y no existe respaldo local en "
            f"{fallback_file}. Coloque una copia de respaldo documentada "
            f"o restaure la conexión."
        )
    logger.warning(
        "Usando copia de RESPALDO LOCAL (%s). Esto debe ser la excepción, "
        "no la regla — documentar en el README por qué se usó.",
        fallback_file,
    )
    return pd.read_csv(fallback_file)


# --------------------------------------------------------------------------
# Validación estructural mínima (previa a cualquier Data Quality Gate)
# --------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas tras la ingesta: {missing_cols}")
    if len(df) < EXPECTED_MIN_ROWS:
        raise ValueError(
            f"El dataset ingerido tiene {len(df)} filas, menos de las "
            f"{EXPECTED_MIN_ROWS} esperadas. Posible descarga incompleta."
        )
    logger.info("Validación estructural de ingesta: OK (%s filas, %s columnas).",
                len(df), len(df.columns))


def compute_checksum(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# --------------------------------------------------------------------------
# Orquestación principal
# --------------------------------------------------------------------------
def ingest() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    source_used = None
    df = None

    for method_name, method in [
        ("ucimlrepo", fetch_via_ucimlrepo),
        ("direct_url", fetch_via_direct_url),
        ("local_fallback", fetch_via_local_fallback),
    ]:
        try:
            df = method()
            source_used = method_name
            break
        except Exception as exc:  # noqa: BLE001 - queremos degradar con gracia
            logger.warning("Falló método de ingesta '%s': %s", method_name, exc)

    if df is None:
        logger.error("Todos los métodos de ingesta fallaron. Abortando.")
        sys.exit(1)

    validate_schema(df)

    df.to_csv(OUTPUT_PATH, index=False)
    checksum = compute_checksum(OUTPUT_PATH)

    metadata = {
        "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_method": source_used,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "sha256_checksum": checksum,
        "output_path": str(OUTPUT_PATH),
        "data_version": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Ingesta completada. Fuente usada: %s", source_used)
    logger.info("Archivo guardado en: %s", OUTPUT_PATH)
    logger.info("Metadata de ingesta guardada en: %s", METADATA_PATH)
    logger.info("data_version = %s (usar este valor en MLflow como parámetro)",
                metadata["data_version"])

    return OUTPUT_PATH


if __name__ == "__main__":
    ingest()
