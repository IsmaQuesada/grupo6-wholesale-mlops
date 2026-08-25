"""
build_features.py — Feature engineering reutilizable para Wholesale Customers.

Este módulo es la ÚNICA fuente de verdad para las transformaciones de
features del proyecto. Se importa igual desde:
  - notebooks/02_eda_feature_engineering.ipynb (exploración)
  - src/training/train.py (producción)

Esto evita el antipatrón que prohíbe la sección I del enunciado: tener
una lógica de features en el notebook y otra distinta en producción.

Uso:
    from src.features.build_features import FeatureBuilder

    fb = FeatureBuilder()
    df_features = fb.fit_transform(df_raw)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Nota: COLS_GASTO se replica en src/data_quality/validate.py (gates de
# calidad). Mantener sincronizados si cambia el esquema del dataset.
COLS_GASTO = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
COLS_PERECEDEROS = ["Fresh", "Frozen", "Delicassen"]
COLS_NO_PERECEDEROS = ["Milk", "Grocery", "Detergents_Paper"]

# Número de componentes principales: menor k con ≥80% de varianza acumulada,
# justificado con evidencia en notebooks/02_eda_feature_engineering.ipynb
# (Análisis 5: k=5 retiene 85.7%; k=3 solo 70.3%).
N_PCA_COMPONENTS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [build_features] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Encapsula todas las transformaciones de feature engineering del
    proyecto, de forma reutilizable y con estado ajustable (fit/transform),
    igual que un transformer de scikit-learn.

    Transformaciones incluidas (justificadas en el notebook de EDA):
      1. Log-transform de las variables de gasto (corrige skewness alto).
      2. Proporciones de gasto por categoría (perfil de composición,
         independiente del volumen total).
      3. Ratio perecederos / no perecederos (separa tipo de negocio).
      4. Índice de diversificación (entropía de Shannon sobre las
         proporciones de gasto).
      5. Escalado estándar de las variables numéricas resultantes.
      6. (Opcional) Reducción a componentes principales vía PCA.
    """

    def __init__(self, cols_gasto: list = None, n_pca_components: int | None = None):
        self.cols_gasto = cols_gasto or COLS_GASTO
        self.n_pca_components = n_pca_components

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components) if n_pca_components else None
        self._fitted = False

    # ------------------------------------------------------------------
    # Transformaciones individuales (funciones puras, testeables)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Log1p sobre variables de gasto para corregir skewness alto."""
        out = df.copy()
        for col in cols:
            out[f"log_{col}"] = np.log1p(out[col])
        return out

    @staticmethod
    def _proporciones_gasto(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Proporción de cada categoría sobre el gasto total del cliente."""
        out = df.copy()
        total = out[cols].sum(axis=1)
        # Evita división por cero (no debería ocurrir tras las gates,
        # pero se protege por robustez ante nuevos datos en producción).
        total_seguro = total.replace(0, np.nan)
        for col in cols:
            out[f"pct_{col}"] = (out[col] / total_seguro).fillna(0)
        out["gasto_total"] = total
        return out

    @staticmethod
    def _ratio_perecederos(df: pd.DataFrame) -> pd.DataFrame:
        """Ratio gasto perecederos / no perecederos (perfil de negocio)."""
        out = df.copy()
        perecederos = out[COLS_PERECEDEROS].sum(axis=1)
        no_perecederos = out[COLS_NO_PERECEDEROS].sum(axis=1).replace(0, np.nan)
        out["ratio_perecederos_no_perecederos"] = (
            perecederos / no_perecederos
        ).fillna(0)
        return out

    @staticmethod
    def _indice_diversificacion(df: pd.DataFrame, cols_pct: list) -> pd.DataFrame:
        """
        Entropía de Shannon sobre las proporciones de gasto: mide qué tan
        concentrado (bajo) o diversificado (alto) es el patrón de compra
        de cada cliente.
        """
        out = df.copy()
        props = out[cols_pct].values
        props_seguro = np.where(props > 0, props, 1e-9)  # evita log(0)
        entropia = -np.sum(props * np.log(props_seguro), axis=1)
        out["indice_diversificacion"] = entropia
        return out

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def _construir_features_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas las transformaciones no paramétricas (sin fit)."""
        out = df.copy()
        out = self._log_transform(out, self.cols_gasto)
        out = self._proporciones_gasto(out, self.cols_gasto)
        out = self._ratio_perecederos(out)
        cols_pct = [f"pct_{c}" for c in self.cols_gasto]
        out = self._indice_diversificacion(out, cols_pct)
        return out

    def _columnas_para_modelo(self, df_features: pd.DataFrame) -> list:
        """Columnas finales que entran al escalado/PCA/modelo de clustering."""
        cols_log = [f"log_{c}" for c in self.cols_gasto]
        cols_pct = [f"pct_{c}" for c in self.cols_gasto]
        extra = ["ratio_perecederos_no_perecederos", "indice_diversificacion"]
        return cols_log + cols_pct + extra

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta el scaler/PCA sobre df y devuelve las features finales."""
        df_features = self._construir_features_base(df)
        cols_modelo = self._columnas_para_modelo(df_features)

        X_scaled = self.scaler.fit_transform(df_features[cols_modelo])

        if self.pca is not None:
            X_final = self.pca.fit_transform(X_scaled)
            cols_finales = [f"PC{i+1}" for i in range(X_final.shape[1])]
        else:
            X_final = X_scaled
            cols_finales = cols_modelo

        self._fitted = True
        self.cols_modelo_ = cols_modelo
        self.cols_finales_ = cols_finales

        resultado = pd.DataFrame(X_final, columns=cols_finales, index=df.index)
        # Se agregan columnas originales/derivadas útiles para interpretar
        # los clusters después (no entran al modelo, son para perfilado).
        resultado_completo = pd.concat(
            [df_features[["Channel", "Region"] + self.cols_gasto + ["gasto_total"]],
             resultado],
            axis=1,
        )
        return resultado_completo

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones ya ajustadas (usar en producción/inferencia
        sobre datos nuevos, nunca reajustar el scaler/PCA con datos nuevos).
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureBuilder no ha sido ajustado. Llama a fit_transform() "
                "primero sobre el set de entrenamiento."
            )
        df_features = self._construir_features_base(df)
        X_scaled = self.scaler.transform(df_features[self.cols_modelo_])

        if self.pca is not None:
            X_final = self.pca.transform(X_scaled)
        else:
            X_final = X_scaled

        resultado = pd.DataFrame(X_final, columns=self.cols_finales_, index=df.index)
        resultado_completo = pd.concat(
            [df_features[["Channel", "Region"] + self.cols_gasto + ["gasto_total"]],
             resultado],
            axis=1,
        )
        return resultado_completo

    # ------------------------------------------------------------------
    # Persistencia del builder ajustado (para serving / inferencia)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """
        Serializa el builder completo (transformaciones + scaler/PCA
        ajustados) con joblib. El archivo resultante es el que cargará la
        API de inferencia para transformar datos nuevos con exactamente
        las mismas transformaciones.
        """
        if not self._fitted:
            raise RuntimeError(
                "Nada que guardar: el builder no ha sido ajustado. "
                "Llama primero a fit_transform()."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FeatureBuilder serializado en %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBuilder":
        """Carga un builder previamente serializado con save()."""
        return joblib.load(path)


def _sha256(path: Path) -> str:
    """Checksum SHA-256 de un archivo (trazabilidad)."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# --------------------------------------------------------------------------
# CLI: genera y persiste las features de forma reproducible e idempotente
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[2]
    RAW_PATH = ROOT_DIR / "data" / "raw" / "wholesale_customers_raw.csv"
    INGEST_METADATA_PATH = ROOT_DIR / "data" / "raw" / "ingestion_metadata.json"
    PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    FEATURES_PATH = PROCESSED_DIR / "wholesale_features.csv"
    METADATA_PATH = PROCESSED_DIR / "features_metadata.json"

    if not RAW_PATH.exists():
        logger.error(
            "No se encontró %s. Corre primero: python src/ingestion/ingest.py",
            RAW_PATH,
        )
        sys.exit(1)

    df_raw = pd.read_csv(RAW_PATH)

    builder = FeatureBuilder(n_pca_components=N_PCA_COMPONENTS)
    df_features = builder.fit_transform(df_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(FEATURES_PATH, index=False)
    builder.save(ROOT_DIR / "models" / "feature_builder.joblib")

    # Trazabilidad: heredar data_version/checksum de la ingesta (usado
    # luego como parámetro en MLflow) y añadir los propios.
    metadata = {
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_raw_csv": str(RAW_PATH),
        "source_sha256": _sha256(RAW_PATH),
        "ingestion_data_version": None,
        "n_rows": len(df_features),
        "n_columns": len(df_features.columns),
        "columns": list(df_features.columns),
        "feature_config": {
            "cols_gasto": list(builder.cols_gasto),
            "n_pca_components": builder.n_pca_components,
            "cols_modelo": list(builder.cols_modelo_),
            "cols_finales": list(builder.cols_finales_),
            "scaler": type(builder.scaler).__name__,
        },
        "output_csv": str(FEATURES_PATH),
        "output_sha256": _sha256(FEATURES_PATH),
    }
    if INGEST_METADATA_PATH.exists():
        with open(INGEST_METADATA_PATH, encoding="utf-8") as f:
            ingestion_meta = json.load(f)
        metadata["ingestion_data_version"] = ingestion_meta.get("data_version")
        metadata["ingestion_source_method"] = ingestion_meta.get("source_method")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Features generadas: %s (%s filas, %s columnas)",
                FEATURES_PATH, len(df_features), len(df_features.columns))
    logger.info("Metadata de features guardada en: %s", METADATA_PATH)
    logger.info("ingestion_data_version = %s (parámetro MLflow)",
                metadata["ingestion_data_version"])
