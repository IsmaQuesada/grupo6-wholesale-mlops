"""
model_monitor.py — Model Monitoring para clustering (Sección O3).

El enunciado pide, para problemas de clustering: distribución de clusters,
desplazamiento de centroides y/o estabilidad del clustering. Este módulo
cubre distribución de clusters y estabilidad (vía comparación de
distribuciones entre referencia y producción) — el desplazamiento de
centroides no aplica aquí porque KMeans no se reentrena en cada batch de
producción; los centroides solo cambian cuando corre train.py de nuevo,
momento que ya queda trazado en MLflow (Sección J).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def cargar_modelo():
    """Carga el modelo K-Means y el FeatureBuilder de producción (mismos artefactos que usa la API)."""
    modelo = joblib.load(REPO_ROOT / "models" / "kmeans_production.joblib")
    feature_builder = joblib.load(REPO_ROOT / "models" / "feature_builder.joblib")
    return modelo, feature_builder


def obtener_distribucion_clusters(modelo, feature_builder, df: pd.DataFrame) -> dict:
    """Predice clusters sobre `df` y retorna el % de registros que cae en cada uno."""
    df_features = feature_builder.transform(df)
    X = df_features[feature_builder.cols_finales_]
    predicciones = modelo.predict(X)

    total = len(predicciones)
    return {
        int(cluster_id): round(int(np.sum(predicciones == cluster_id)) / total * 100, 2)
        for cluster_id in sorted(set(predicciones))
    }


def comparar_distribuciones(dist_referencia: dict, dist_produccion: dict, umbral: float = 10.0) -> dict:
    """
    Compara dos distribuciones de clusters (en % por cluster).

    umbral=10.0 significa: si algún cluster gana o pierde más de 10 puntos
    porcentuales de participación entre referencia y producción, se marca
    como inestable. 10 puntos es un punto de partida razonable dado que los
    3 clusters de este proyecto ya están relativamente balanceados
    (30-38% cada uno, ver notebook 03); un desbalance de 10pp movería a
    algún cluster fuera de ese rango típico.
    """
    todos_los_clusters = sorted(set(dist_referencia) | set(dist_produccion))

    diferencias = {
        f"cluster_{c}": round(abs(dist_referencia.get(c, 0) - dist_produccion.get(c, 0)), 2)
        for c in todos_los_clusters
    }
    diferencia_maxima = max(diferencias.values()) if diferencias else 0

    return {
        "estable": diferencia_maxima <= umbral,
        "diferencias": diferencias,
        "diferencia_maxima": diferencia_maxima,
    }