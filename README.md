# Grupo 6 — MLOps End-to-End: Segmentación de Clientes Mayoristas (Wholesale Customers)

> Tipo de problema: **Clustering (no supervisado)**
> Dataset: [Wholesale Customers — UCI ML Repository (ID 292)](https://archive.ics.uci.edu/dataset/292/wholesale+customers)

---

## 1. Business Problem

El dataset representa el gasto anual (en unidades monetarias) de clientes de
un distribuidor mayorista, distribuido en 6 categorías de producto (Fresh,
Milk, Grocery, Frozen, Detergents_Paper, Delicassen).

## 2. Dataset

- **Fuente:** UCI Machine Learning Repository, dataset ID 292.
- **Tamaño:** 440 registros, 8 columnas, sin valores faltantes.
- **Licencia:** CC BY 4.0.
- **Nota importante:** el CSV crudo **no se sube directamente** como fuente
  de verdad del proyecto. Se obtiene mediante
  `src/ingestion/ingest.py`. Una copia pequeña de respaldo
  (`data/raw/_fallback/`) se mantiene en el repo únicamente como plan de
  contingencia para demos sin conexión a internet.

## 3. Architecture

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
│       └── _fallback/          # copia pequeña de respaldo (offline demo)
└── src/
    └── ingestion/
        └── ingest.py           # ✅ implementado
```

**Próximas carpetas a agregar según se completen esas etapas del
enunciado:** `src/data_quality/`, `src/features/`, `src/training/`,
`src/monitoring/`, `src/api/`, `notebooks/`, `tests/`, `models/`,
`monitoring_reports/`, `docs/`.

## 5. Installation

```bash
git clone https://github.com/<usuario>/grupo6-wholesale-mlops.git
cd grupo6-wholesale-mlops

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 6. Data Ingestion

Script reproducible: `src/ingestion/ingest.py`

```bash
python src/ingestion/ingest.py
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
`src/data_quality/validate.py` (próxima entrega).

## 7. Training

_(pendiente — próxima entrega)_

## 8. MLflow

_(pendiente — próxima entrega)_

## 9. Docker

_(pendiente — próxima entrega)_

## 10. API

_(pendiente — próxima entrega)_

## 11. Monitoring

_(pendiente — próxima entrega)_

## 12. Results

_(pendiente — próxima entrega)_
