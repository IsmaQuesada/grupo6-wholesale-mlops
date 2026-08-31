"""
validate.py — Data Quality Gates para el dataset Wholesale Customers.

Ejecuta reglas automáticas de calidad ANTES de que el pipeline avance a
feature engineering / entrenamiento. Corresponde al bloque
"DATA VALIDATION -> PASS/FAIL -> ALERT" de la arquitectura MLOps del
proyecto (sección G del enunciado).

Las reglas y sus umbrales fueron definidos y justificados en el
diagnóstico exploratorio: notebooks/01_data_quality_diagnostico.ipynb

Uso como script:
    python src/data_quality/validate.py

Uso como módulo (desde train.py u otro script del pipeline):
    from src.data_quality.validate import validar_calidad_datos
    reporte = validar_calidad_datos(df, cols_gasto)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import COLS_GASTO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [data_quality] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Configuración de las gates (umbrales justificados en el notebook)
# --------------------------------------------------------------------
N_FILAS_HISTORICO = 440  # tamaño del dataset original en UCI ML Repository
TOLERANCIA_MIN_FILAS = 0.9  # se acepta hasta un 10% menos que el histórico
UMBRAL_DUPLICADOS = 0.02  # 2% máximo de filas duplicadas

CHANNEL_VALIDOS = {1, 2}
REGION_VALIDOS = {1, 2, 3}

COLUMNAS_ESPERADAS = set(COLS_GASTO) | {"Channel", "Region"}


def validar_calidad_datos(data: pd.DataFrame, cols_gasto: list = None) -> dict:
    """
    Ejecuta las Data Quality Gates sobre un DataFrame de Wholesale Customers.

    Devuelve un diccionario con el resultado (PASS/FAIL) de cada regla,
    junto con el detalle correspondiente para trazabilidad.
    """
    if cols_gasto is None:
        cols_gasto = COLS_GASTO

    resultados = {}

    # Regla 1: mínimo de filas esperado (detecta descargas incompletas
    # o cambios drásticos de volumen en producción)
    min_filas_esperadas = int(TOLERANCIA_MIN_FILAS * N_FILAS_HISTORICO)
    resultados["min_rows"] = {
        "pass": len(data) >= min_filas_esperadas,
        "detalle": (
            f"{len(data)} filas (esperado: >= {min_filas_esperadas}, "
            f"{int(TOLERANCIA_MIN_FILAS * 100)}% del histórico)"
        ),
    }

    # Regla 2: sin nulos en columnas obligatorias
    columnas_obligatorias = cols_gasto + ["Channel", "Region"]
    nulos = data[columnas_obligatorias].isna().sum().sum()
    resultados["sin_nulos_obligatorios"] = {
        "pass": nulos == 0,
        "detalle": f"{nulos} valores nulos en columnas obligatorias",
    }

    # Regla 3: duplicados bajo umbral
    prop_duplicados = data.duplicated().mean()
    resultados["duplicados_bajo_umbral"] = {
        "pass": prop_duplicados < UMBRAL_DUPLICADOS,
        "detalle": f"{prop_duplicados:.2%} de filas duplicadas (umbral: {UMBRAL_DUPLICADOS:.0%})",
    }

    # Regla 4: tipos numéricos válidos — se evalúa ANTES de cualquier comparación
    # aritmética (Regla 5), porque una columna de gasto con datos no numéricos
    # (ej. un string) haría fallar esa comparación y detendría todo el proceso.
    columnas_no_numericas = [c for c in cols_gasto if not pd.api.types.is_numeric_dtype(data[c])]
    resultados["tipos_numericos_validos"] = {
        "pass": len(columnas_no_numericas) == 0,
        "detalle": (
            f"Columnas con tipo no numérico: {columnas_no_numericas}"
            if columnas_no_numericas else "Todas las columnas de gasto son numéricas"
        ),
    }

    # Regla 5: sin gastos negativos — solo se evalúa sobre columnas ya confirmadas
    # numéricas por la Regla 4; una columna con tipo inválido queda excluida aquí
    # para no volver a intentar una comparación que rompería el pipeline.
    cols_gasto_numericas = [c for c in cols_gasto if c not in columnas_no_numericas]
    negativos = (data[cols_gasto_numericas] < 0).sum().sum() if cols_gasto_numericas else 0
    resultados["sin_gastos_negativos"] = {
        "pass": negativos == 0,
        "detalle": f"{negativos} valores negativos en variables de gasto",
    }

    # Regla 6: cardinalidad categórica válida (Channel/Region dentro de lo esperado)
    channel_valido = set(data["Channel"].unique()).issubset(CHANNEL_VALIDOS)
    region_valida = set(data["Region"].unique()).issubset(REGION_VALIDOS)
    resultados["cardinalidad_categorica_valida"] = {
        "pass": channel_valido and region_valida,
        "detalle": f"Channel válido: {channel_valido}, Region válido: {region_valida}",
    }

    # Regla 7: esquema sin columnas inesperadas (detecta modificaciones de esquema en el batch)
    columnas_extra = set(data.columns) - COLUMNAS_ESPERADAS
    resultados["esquema_sin_columnas_extra"] = {
        "pass": len(columnas_extra) == 0,
        "detalle": (
            f"Columnas inesperadas: {sorted(columnas_extra)}"
            if columnas_extra else "Sin columnas fuera del esquema esperado"
        ),
    }

    return resultados


def imprimir_reporte(reporte: dict) -> bool:
    """Imprime el reporte de forma legible y devuelve True si todo pasó."""
    for regla, resultado in reporte.items():
        estado = "✅ PASS" if resultado["pass"] else "❌ FAIL"
        logger.info("%s | %s: %s", estado, regla, resultado["detalle"])

    todas_pasaron = all(r["pass"] for r in reporte.values())
    if todas_pasaron:
        logger.info("Resultado global: PASS ✅ — el pipeline puede continuar.")
    else:
        logger.error("Resultado global: FAIL ❌ — revisar reglas marcadas arriba.")
    return todas_pasaron


if __name__ == "__main__":
    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    CSV_PATH = RAW_DIR / "wholesale_customers_raw.csv"

    if not CSV_PATH.exists():
        logger.error(
            "No se encontró %s. Corre primero: python src/ingestion/ingest.py",
            CSV_PATH,
        )
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    reporte = validar_calidad_datos(df, COLS_GASTO)
    ok = imprimir_reporte(reporte)

    if not ok:
        sys.exit(1)  # detiene el pipeline si alguna gate crítica falla
