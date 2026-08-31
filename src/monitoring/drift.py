"""
drift.py — Detección de data drift (Sección O2) usando PSI (Population
Stability Index): compara la distribución de referencia P_reference(X)
contra la distribución de producción P_production(X) para cada variable.

Los umbrales de clasificación (0.10 / 0.25) son los valores de referencia
estándar de la literatura de PSI, usados como punto de partida. El
enunciado exige que los thresholds se justifiquen con evidencia propia,
no que se traten como ley universal — esa justificación se hace con datos
reales del proyecto en la simulación de drift.
"""
import numpy as np
import pandas as pd


def calcular_psi(referencia: pd.Series, produccion: pd.Series, bins: int = 10) -> float:
    """Calcula el PSI entre dos distribuciones de una misma variable."""
    referencia = np.array(referencia)
    produccion = np.array(produccion)

    edges = np.histogram_bin_edges(referencia, bins=bins)
    ref_counts, _ = np.histogram(referencia, bins=edges)
    prod_counts, _ = np.histogram(produccion, bins=edges)

    # +1 en cada bin (suavizado de Laplace): evita división por cero y
    # log(0) si algún bin de referencia o producción queda vacío.
    ref_pct = (ref_counts + 1) / (len(referencia) + bins)
    prod_pct = (prod_counts + 1) / (len(produccion) + bins)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


def calcular_psi_dataframe(
    df_referencia: pd.DataFrame, df_produccion: pd.DataFrame, columnas: list
) -> dict:
    """Calcula el PSI de cada columna en `columnas` presente en ambos DataFrames."""
    resultados = {}
    for col in columnas:
        if col in df_referencia.columns and col in df_produccion.columns:
            resultados[col] = round(
                calcular_psi(df_referencia[col].dropna(), df_produccion[col].dropna()), 4
            )
    return resultados


def clasificar_psi(valor_psi: float) -> str:
    """Clasifica un valor de PSI según los umbrales estándar de referencia."""
    if valor_psi < 0.10:
        return "OK"
    elif valor_psi < 0.25:
        return "WARNING"
    else:
        return "ALERT"