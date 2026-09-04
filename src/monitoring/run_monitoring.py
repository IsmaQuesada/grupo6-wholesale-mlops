"""
run_monitoring.py — Script ejecutable de monitoreo (Sección O).

Ejecuta las tres dimensiones de monitoreo sobre datos de referencia vs.
producción y devuelve una decisión de reentrenamiento:

    O1: System Monitoring  — métricas operativas del servicio API
    O2: Data Monitoring    — PSI (Population Stability Index) por variable
    O3: Model Monitoring   — distribución y estabilidad de clusters

Uso:
    python src/monitoring/run_monitoring.py [ruta_csv_produccion]

Si no se proporciona ruta, se usa el CSV de referencia como producción
(para demostración / smoke test).
"""
import json
import sys
from pathlib import Path

# Agregar raíz del repo al path para imports de src.*
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.features.build_features import COLS_GASTO
from src.monitoring.drift import calcular_psi_dataframe, clasificar_psi
from src.monitoring.model_monitor import (
    cargar_modelo,
    comparar_distribuciones,
    obtener_distribucion_clusters,
)
from src.monitoring.retrain_trigger import evaluar_necesidad_reentrenamiento
from src.monitoring.system_metrics import get_metrics


def _separator(titulo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


def main() -> int:
    # --- Datos ---
    ref_path = REPO_ROOT / "data" / "raw" / "wholesale_customers_raw.csv"
    if not ref_path.exists():
        print(f"ERROR: No se encontró {ref_path}")
        print("  Ejecuta primero: python src/ingestion/ingest.py")
        return 2

    df_ref = pd.read_csv(ref_path)
    print(f"Datos de referencia: {len(df_ref)} filas ({ref_path.name})")

    if len(sys.argv) > 1:
        prod_path = Path(sys.argv[1])
        if not prod_path.exists():
            print(f"ERROR: No se encontró {prod_path}")
            return 2
        df_prod = pd.read_csv(prod_path)
        print(f"Datos de producción:  {len(df_prod)} filas ({prod_path.name})")
    else:
        df_prod = df_ref
        print("Datos de producción:  mismos que referencia (smoke test)")

    # --- Cargar modelo ---
    try:
        modelo, fb = cargar_modelo()
    except FileNotFoundError:
        print("\nERROR: No se encontraron los artefactos del modelo.")
        print("  Ejecuta primero: python src/training/train.py")
        return 2

    # --- O1: System Monitoring ---
    _separator("O1 — System Monitoring")
    metrics = get_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # --- O2: Data Monitoring (PSI) ---
    _separator("O2 — Data Monitoring (PSI)")
    psi_por_variable = calcular_psi_dataframe(df_ref, df_prod, COLS_GASTO)
    psi_max = max(psi_por_variable.values())
    print(f"  {'Variable':<20} {'PSI':>8}  Clasificación")
    print(f"  {'-'*20} {'-'*8}  {'-'*15}")
    for var, psi_val in psi_por_variable.items():
        clasif = clasificar_psi(psi_val)
        print(f"  {var:<20} {psi_val:>8.4f}  {clasif}")
    print(f"\n  PSI maximo: {psi_max:.4f} -> {clasificar_psi(psi_max)}")

    # --- O3: Model Monitoring ---
    _separator("O3 — Model Monitoring (clusters)")
    dist_ref = obtener_distribucion_clusters(modelo, fb, df_ref)
    dist_prod = obtener_distribucion_clusters(modelo, fb, df_prod)
    comparacion = comparar_distribuciones(dist_ref, dist_prod)

    print(f"  {'Cluster':<10} {'Ref %':>8}  {'Prod %':>8}  {'Diff pp':>8}")
    print(f"  {'-'*10} {'-'*8}  {'-'*8}  {'-'*8}")
    for c in sorted(set(dist_ref) | set(dist_prod)):
        ref_v = dist_ref.get(c, 0)
        prod_v = dist_prod.get(c, 0)
        diff = abs(ref_v - prod_v)
        print(f"  Cluster {c:<3} {ref_v:>7.2f}%  {prod_v:>7.2f}%  {diff:>7.2f}")
    print(f"\n  Diferencia maxima: {comparacion['diferencia_maxima']:.2f}pp -> "
          f"{'Estable' if comparacion['estable'] else 'Inestable'}")

    # --- Retrain Trigger ---
    _separator("R — Evaluación de reentrenamiento")
    metadata_path = REPO_ROOT / "models" / "production_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        silhouette_actual = json.load(f)["silhouette"]
    resultado = evaluar_necesidad_reentrenamiento(
        df_ref, df_prod, silhouette_actual, modelo, fb,
    )
    print(f"  Decisión: {resultado['decision']}")
    print(f"  Detalles:")
    print(f"    psi_max:                {resultado['psi_max']}")
    print(f"    silhouette_actual:      {resultado['silhouette_actual']}")
    print(f"    umbral_silhouette:      {resultado['umbral_silhouette']}")
    print(f"    dif_composición:        {resultado['diferencia_composicion_pp']:.2f}pp")
    print(f"    drift_alto:             {resultado['drift_alto']}")
    print(f"    performance_baja:       {resultado['performance_baja']}")
    print(f"    modelo_inestable:       {resultado['modelo_inestable']}")

    # --- Resumen ---
    _separator("Resumen")
    print(f"  PSI máximo:      {psi_max:.4f} ({clasificar_psi(psi_max)})")
    print(f"  Composición:     {'Estable' if comparacion['estable'] else 'Inestable'}")
    print(f"  Decisión:        {resultado['decision']}")
    print()

    if resultado["decision"] == "REENTRENAR":
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
