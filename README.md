# Grupo 6 — MLOps End-to-End: Segmentación de Clientes Mayoristas (Wholesale Customers)

> Curso: MLOps End-to-End | Proyecto (60%)
> Tipo de problema: **Clustering (no supervisado)**
> Dataset: [Wholesale Customers — UCI ML Repository (ID 292)](https://archive.ics.uci.edu/dataset/292/wholesale+customers)

---

## 1. Business Problem

El dataset representa el gasto anual (en unidades monetarias) de clientes de
un distribuidor mayorista, distribuido en 6 categorías de producto (Fresh,
Milk, Grocery, Frozen, Detergents_Paper, Delicassen).

El objetivo del proyecto es **descubrir estructuras/segmentos de clientes
nuevos** a partir únicamente del comportamiento de gasto, **sin usar
inicialmente** las columnas `Channel` (canal Horeca/Retail) ni `Region`
como variables de entrenamiento. Estas dos columnas se usan únicamente
al final, como validación externa, para verificar si los segmentos
descubiertos capturan información más rica que la ya conocida.

## 2. Dataset

- **Fuente:** UCI Machine Learning Repository, dataset ID 292.
- **Tamaño:** 440 registros, 8 columnas, sin valores faltantes.
- **Licencia:** CC BY 4.0.
- **Nota importante:** el CSV crudo **no se sube directamente** como fuente
  de verdad del proyecto (sección D del enunciado). Se obtiene mediante
  `src/ingestion/ingest.py`. Una copia pequeña de respaldo
  (`data/raw/_fallback/`) se mantiene en el repo únicamente como plan de
  contingencia para demos sin conexión a internet — ver sección 6.

## 3. Architecture

Ver diagrama completo en `docs/architecture.png` y su justificación en
`docs/architecture.md` (se agregará en una entrega posterior). Arquitectura
general obligatoria del curso:

```
Fuente de datos → Data Ingestion → Raw/Bronze → Data Validation
   → (FAIL → Alert | PASS → Data Cleaning) → Feature Pipeline
   → Training → Evaluation → MLflow (Tracking + Registry) → Best Candidate
   → Dockerize → Model API → Production → Monitoring
   (Data Drift / Model Performance / System Metrics) → Retrain Trigger
```

## 4. Repository Structure

Estructura actual (se irá ampliando conforme avancen las etapas del
proyecto; no se crean carpetas vacías por adelantado):

```
grupo6-wholesale-mlops/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       ├── _fallback/                      # copia pequeña de respaldo (offline demo)
│       ├── wholesale_customers_raw.csv     # generado por ingest.py (no versionado)
│       └── ingestion_metadata.json         # generado por ingest.py (no versionado)
├── notebooks/
│   └── 01_data_quality_diagnostico.ipynb   # ✅ implementado
└── src/
    ├── ingestion/
    │   ├── ingest.py                       # ✅ implementado
    └── data_quality/
        └── validate.py                     # ✅ implementado
```

**Próximas carpetas a agregar según se completen esas etapas del
enunciado:** `src/features/`, `src/training/`, `src/monitoring/`,
`src/api/`, `tests/`, `models/`, `monitoring_reports/`, `docs/`.

## 5. Installation

```bash
git clone https://github.com/<usuario>/grupo6-wholesale-mlops.git
cd grupo6-wholesale-mlops

python -m venv venv o py -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 6. Data Ingestion

Script reproducible: `src/ingestion/ingest.py`

```bash
python src/ingestion/ingest.py o py src/ingestion/ingest.py
```

**Estrategia de ingesta (en orden de intento):**

1. Paquete oficial `ucimlrepo` (`fetch_ucirepo(id=292)`).
2. Descarga directa del CSV publicado por UCI vía URL.
3. Copia local de respaldo (`data/raw/_fallback/`) — **solo** si no hay
   conexión a internet disponible. El script emite un `WARNING` explícito
   cuando usa este método, para dejar evidencia clara de que fue la
   excepción y no la fuente primaria.

**Salidas del script:**

- `data/raw/wholesale_customers_raw.csv` — dataset crudo ingerido (no se
  versiona en git, se regenera en cada ejecución).
- `data/raw/ingestion_metadata.json` — metadata de trazabilidad: método de
  ingesta usado, timestamp UTC, checksum SHA-256, número de filas/columnas
  y un `data_version` (usado luego como parámetro en MLflow).

**Validación mínima incluida:** el script verifica que las 8 columnas
esperadas existan y que el dataset tenga al menos 400 filas antes de
aceptar la ingesta como válida. Esta es la primera capa de validación;
las reglas de calidad más profundas se implementan en
`src/data_quality/validate.py` (documentado en la siguiente sección).

## 7. Data Quality

**Diagnóstico exploratorio:** `notebooks/01_data_quality_diagnostico.ipynb`

El notebook investiga, con evidencia y justificación para cada decisión
(no transformaciones aplicadas "a ciegas"):

- Valores faltantes explícitos y codificados (ej. ceros/negativos sospechosos)
- Duplicados totales y duplicados parciales por perfil de gasto
- Cardinalidad y validez de `Channel` / `Region` frente a lo documentado en UCI
- Valores extremos y datos imposibles (gastos negativos)
- Skewness y correlación excesiva entre variables de gasto

**Hallazgos principales:**

- El dataset **no presenta** nulos, duplicados, ni valores negativos.
- Los valores extremos detectados (ej. en `Delicassen`, `Frozen`)
  corresponden a clientes reales de alto volumen de compra, no a errores
  de captura — por lo tanto **no se eliminan**, se tratarán con
  transformación logarítmica en la etapa de Feature Engineering.
- Las seis variables de gasto presentan **skewness alto** (todas por
  encima de 2 en valor absoluto), lo que confirma la necesidad de
  transformar antes de aplicar algoritmos de clustering sensibles a
  escala/distribución como K-Means.

**Data Quality Gates (reglas automáticas):** `src/data_quality/validate.py`

```bash
python src/data_quality/validate.py o py src/data_quality/validate.py
```

Implementa 5 reglas automáticas, cada una con umbral justificado en el
notebook de diagnóstico:

| Regla | Qué valida | Umbral |
|---|---|---|
| `min_rows` | El dataset no perdió filas significativamente respecto al histórico (440) | ≥ 90% del histórico |
| `sin_nulos_obligatorios` | No hay nulos en columnas de gasto ni en Channel/Region | 0 nulos |
| `duplicados_bajo_umbral` | Proporción de filas duplicadas | < 2% |
| `sin_gastos_negativos` | Ninguna variable de gasto tiene valores negativos (dato imposible) | 0 negativos |
| `cardinalidad_categorica_valida` | Channel ∈ {1,2} y Region ∈ {1,2,3} | sin categorías inesperadas |

Este script está diseñado para ejecutarse tanto de forma independiente
(`python src/data_quality/validate.py`) como importado desde otros
scripts del pipeline (`src/training/train.py`, próxima entrega), para
que la validación ocurra automáticamente antes de entrenar, sin
depender de que alguien la corra manualmente desde el notebook.

## 8. Training
_(pendiente — próxima entrega)_

## 9. MLflow
_(pendiente — próxima entrega)_

## 10. Docker
_(pendiente — próxima entrega)_

## 11. API
_(pendiente — próxima entrega)_

## 12. Monitoring
_(pendiente — próxima entrega)_

## 13. Results
_(pendiente — próxima entrega)_

## 14. Team

| Integrantes |
|---|
| Ismael Quesada Salas | 
| Magaly Bushey Ventura | 
