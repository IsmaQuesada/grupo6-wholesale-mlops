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
├── Dockerfile
├── .dockerignore
├── mlflow.db                                # generado por train.py/notebook 03 (no versionado)
│
├── data/
│   ├── raw/
│   │   ├── _fallback/                      # copia pequeña de respaldo (offline demo)
│   │   ├── wholesale_customers_raw.csv     # generado por ingest.py (no versionado)
│   │   └── ingestion_metadata.json         # generado por ingest.py (no versionado)
│   └── processed/
│       ├── wholesale_features.csv          # generado por build_features.py (no versionado)
│       └── features_metadata.json          # generado por build_features.py (no versionado)
│
├── models/
│   ├── kmeans_production.joblib            # generado por train.py en promoción a Production (no versionado)
│   ├── feature_builder.joblib              # generado por build_features.py / train.py (no versionado)
│   └── production_metadata.json            # generado por train.py en promoción a Production (no versionado)
│
├── notebooks/
│   ├── 01_data_quality_diagnostico.ipynb          # ✅ implementado (sección F)
│   ├── 02_eda_feature_engineering.ipynb           # ✅ implementado (secciones H, I)
│   └── 03_modelado_experiment_tracking.ipynb      # ✅ implementado (secciones J, K)
│
└── src/
    ├── ingestion/
    │   └── ingest.py                       # ✅ implementado
    ├── data_quality/
    │   └── validate.py                     # ✅ implementado
    ├── features/
    │   └── build_features.py               # ✅ implementado
    ├── training/
    │   └── train.py                        # ✅ implementado (secciones J, K)
    └── api/
        ├── main.py                         # ✅ implementado (FastAPI)
        └── schemas.py                      # ✅ implementado (Pydantic)
```

**Próximas carpetas a agregar según se completen esas etapas del
enunciado:** `src/monitoring/`, `tests/`, `monitoring_reports/`,
`docs/` (más allá de la guía interna ya existente).

## 5. Installation

```bash
git clone https://github.com/<usuario>/grupo6-wholesale-mlops.git
cd grupo6-wholesale-mlops

python -m venv venv        # o: py -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 6. Data Ingestion

Script reproducible: `src/ingestion/ingest.py`

```bash
python src/ingestion/ingest.py     # o: py src/ingestion/ingest.py
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
- Asimetría y correlación excesiva entre variables de gasto

**Hallazgos principales:**

- El dataset **no presenta** nulos, duplicados, ni valores negativos.
- Los valores extremos detectados corresponden a clientes reales de alto
  volumen de compra, no a errores de captura — por lo tanto **no se
  eliminan**, se tratan con transformación logarítmica en la etapa de
  Feature Engineering (ver sección 8).
- Las seis variables de gasto presentan **asimetría alto** (todas por
  encima de 2 en valor absoluto, `Delicassen` la más extrema con 11.15),
  lo que confirma la necesidad de transformar antes de aplicar algoritmos
  de clustering sensibles a escala/distribución como K-Means.

**Data Quality Gates (reglas automáticas):** `src/data_quality/validate.py`

```bash
python src/data_quality/validate.py     # o: py src/data_quality/validate.py
```

Implementa 5 reglas automáticas, cada una con umbral justificado en el
notebook de diagnóstico:

| Regla                            | Qué valida                                                                | Umbral                     |
| -------------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| `min_rows`                       | El dataset no perdió filas significativamente respecto al histórico (440) | ≥ 90% del histórico        |
| `sin_nulos_obligatorios`         | No hay nulos en columnas de gasto ni en Channel/Region                    | 0 nulos                    |
| `duplicados_bajo_umbral`         | Proporción de filas duplicadas                                            | < 2%                       |
| `sin_gastos_negativos`           | Ninguna variable de gasto tiene valores negativos (dato imposible)        | 0 negativos                |
| `cardinalidad_categorica_valida` | Channel ∈ {1,2} y Region ∈ {1,2,3}                                        | sin categorías inesperadas |

Este script está diseñado para ejecutarse tanto de forma independiente
como importado desde otros scripts del pipeline (`src/training/train.py`),
para que la validación ocurra automáticamente antes de entrenar, sin
depender de que alguien la corra manualmente desde el notebook.

## 8. EDA y Feature Engineering

**Análisis exploratorio orientado a decisiones:**
`notebooks/02_eda_feature_engineering.ipynb`

Cada análisis de este notebook responde explícitamente la pregunta que
exige la sección H del enunciado: _¿qué decisión de modelado, limpieza,
ingeniería de variables o negocio cambia como consecuencia de este
resultado?_ Los hallazgos principales:

| Hallazgo                                                                                                                    | Decisión                                                                                                       | Se implementa en                                                            |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Asimetría alto en las 6 variables de gasto (todas > 2, `Delicassen` = 11.15)                                                | Log-transform (log1p)                                                                                          | `FeatureBuilder._log_transform`                                             |
| Correlación alta entre Grocery, Detergents_Paper y Milk (hasta 0.92) + colinealidad perfecta en las proporciones (Σpct = 1) | PCA con k elegido por varianza acumulada ≥80% (k=5, retiene 85.7%)                                             | `FeatureBuilder(n_pca_components=5)`                                        |
| Outliers son clientes reales de alto volumen (`Frozen` el más frecuente, 43 casos), no errores                              | No se eliminan; se comprimen con log-transform                                                                 | `FeatureBuilder._log_transform`                                             |
| Colas ya acotadas tras log1p (máx. \|z\| = 6.36 vs 4.54 con RobustScaler)                                                   | StandardScaler; RobustScaler no es necesario                                                                   | `FeatureBuilder.scaler`                                                     |
| El perfil de composición del gasto es distinto al volumen total                                                             | Proporciones por categoría + ratio perecederos/no perecederos + índice de diversificación (entropía)           | `_proporciones_gasto`, `_ratio_perecederos`, `_indice_diversificacion`      |
| Preview de clusterabilidad: silhouette más alto en k=2 (0.322), posible riesgo de "redescubrir" Channel                     | Confirmado en training (sección 9): k=3 elegido por aportar valor de negocio adicional pese a menor silhouette | `src/training/train.py` y `notebooks/03_modelado_experiment_tracking.ipynb` |
| El enfoque del proyecto pide descubrir estructura sin partir de categorías ya conocidas                                     | No usar Channel/Region como input del clustering; solo validación externa posterior                            | Validación externa hecha en el notebook de modelado (sección 9)             |

**Feature Engineering reutilizable:** `src/features/build_features.py`

```bash
python src/features/build_features.py     # o: py src/features/build_features.py
```

La clase `FeatureBuilder` es la **única fuente de verdad** de las
transformaciones del proyecto — se importa igual desde los notebooks de
EDA/modelado y desde `src/training/train.py`, evitando el antipatrón
de tener una lógica de features en el notebook y otra distinta en
producción (sección I del enunciado). Provee:

- Transformaciones: log-transform, proporciones de gasto, ratio
  perecederos/no perecederos, índice de diversificación, escalado y PCA.
- Interfaz `fit_transform` / `transform` estilo scikit-learn, para nunca
  reajustar el scaler/PCA con datos nuevos en producción.
- Persistencia (`save` / `load` vía `joblib`) del builder ya ajustado, para
  que la futura API de inferencia aplique exactamente las mismas
  transformaciones sobre datos nuevos.

**Salidas del script:**

- `data/processed/wholesale_features.csv` — dataset de features listo para
  entrenamiento (no se versiona, se regenera con el script).
- `data/processed/features_metadata.json` — trazabilidad: configuración de
  features usada, checksums, y el `data_version` heredado de la ingesta
  (parámetro que se registrará en MLflow).
- `models/feature_builder.joblib` — el `FeatureBuilder` ajustado y
  serializado, reutilizable en entrenamiento e inferencia.

## 9. Training

**Exploración y comparación de modelos:**
`notebooks/03_modelado_experiment_tracking.ipynb`

Se compararon **10 configuraciones** (2 algoritmos × k=2 a 6): K-Means y
Clustering Jerárquico (Agglomerative). K-Means dominó consistentemente en
silhouette y Davies-Bouldin en los 6 valores de k probados, por lo que la
comparación final de candidatos se hizo entre **k=2** y **k=3** con K-Means.

**Criterio explícito de selección (sección K):**

| Candidato                 | Silhouette              | Davies-Bouldin | Observación clave                                                                                                                                                           |
| ------------------------- | ----------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K-Means k=2               | **0.322** (el más alto) | 1.231          | Cluster dominante es 96% Channel=Horeca — redescubre casi directamente la variable que el proyecto decidió no usar como input                                               |
| **K-Means k=3 (elegido)** | 0.257 (segundo mejor)   | 1.393          | Introduce un tercer perfil de negocio genuinamente distinto: separa "Horeca fresco-especializado" de "Horeca diversificado", información que Channel por sí solo no captura |

**Se eligió k=3** sobre k=2 pese a su silhouette más bajo, porque aporta
segmentación de negocio más rica y no se limita a redescubrir una variable
ya conocida — cumpliendo el enfoque original del proyecto (sección
"Business Problem"). Ambos candidatos tienen clusters balanceados (sin
grupos degenerados) y fueron validados cruzando contra `Channel`/`Region`
como verificación externa, nunca como input del modelo.

**Validación de estabilidad:** antes de promover el modelo, se re-entrenó
k=3 con una semilla alternativa (123). El silhouette varió de 0.257 a
0.256 (diferencia de 0.000, muy por debajo del umbral de 0.05 definido),
confirmando que el resultado es estable y no depende de la inicialización
aleatoria.

**Reproducción en producción:** `src/training/train.py`

```bash
python src/training/train.py     # o: py src/training/train.py
```

Reproduce en código de producción la configuración ganadora decidida en
el notebook (K-Means, k=3, `K_PCA=5`), sin reimplementar la lógica de
features (importa `FeatureBuilder`) ni la de calidad de datos (importa
`validar_calidad_datos`). Cada ejecución:

1. Carga los datos crudos y valida las Data Quality Gates (detiene el
   pipeline si alguna falla).
2. Construye las features con `FeatureBuilder`.
3. Entrena K-Means (k=3) y calcula silhouette, Davies-Bouldin e inertia.
4. Registra un nuevo run en MLflow con parámetros, métricas y el modelo
   como artifact.
5. Valida estabilidad con una semilla alternativa y, si pasa, promueve el
   modelo a Production en el Model Registry.

## 10. MLflow

**Tracking URI:** SQLite (`mlflow.db`, en la raíz del repo) — se eligió
sobre el file store por defecto porque el Model Registry (sección K)
requiere un backend con soporte completo de versiones de modelo.

**Ver la interfaz de MLflow:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Cada run registra (mínimo exigido por la sección J):**

| Categoría      | Contenido registrado                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Parameters** | `algorithm`, `n_clusters`, `feature_set` (ej. `PCA_5_componentes`), `random_seed`, `data_version` (heredado de `ingestion_metadata.json`)        |
| **Metrics**    | `silhouette`, `davies_bouldin`, `inertia`; en el run final también `silhouette_semilla_alternativa` y `diferencia_estabilidad`                   |
| **Artifacts**  | El modelo entrenado (`mlflow.sklearn.log_model`), además de los gráficos de comparación de clusters y perfiles de gasto generados en el notebook |

**Model Registry — ciclo Experiment → Candidate → Validation → Production:**

1. **Experiment:** los 10 runs de comparación (k=2-6 × 2 algoritmos) quedan
   registrados bajo el experimento `wholesale-clustering-grupo6`.
2. **Candidate:** el run ganador (K-Means, k=3) se registra con
   `mlflow.register_model()` y se etiqueta `lifecycle_stage=candidate`.
3. **Validation:** se re-entrena con una semilla alternativa; si la
   diferencia de silhouette es menor al umbral (0.05), se etiqueta
   `lifecycle_stage=validation`.
4. **Production:** se le asigna el alias `production` con
   `set_registered_model_alias()` (API moderna de MLflow — se evitó
   `transition_model_version_stage()` por estar deprecada desde la
   versión 2.9.0), garantizando que este mecanismo sea idéntico tanto en
   el notebook de exploración como en `src/training/train.py`.

Este ciclo responde directamente la pregunta que exige la sección J:
_¿exactamente qué datos, código, features, hiperparámetros y métricas
produjeron este modelo?_ — toda esa información queda trazada en el run
registrado bajo el alias `production`.

## 11. Docker

El modelo se sirve dentro de un contenedor Docker ligero (sección L del
enunciado). La imagen usa `python:3.11-slim` (~150 MB vs ~900 MB de la
imagen completa) e instala **solo** las dependencias necesarias para la
API de inferencia (sin seaborn, matplotlib, jupyter, etc.).

**Construir la imagen:**

```bash
docker build -t grupo6-mlops .
```

**Ejecutar el contenedor:**

```bash
# Lo mínimo necesario (incluye los artefactos del modelo en la imagen)
docker run -p 8000:8000 grupo6-mlops

# Opcional: montar models/ como volumen para no reconstruir la imagen
# cuando se re-entrene el modelo
docker run -p 8000:8000 -v "${PWD}/models:/app/models" grupo6-mlops
```

**Verificar que funciona:**

```powershell
# PowerShell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"Fresh": 12669, "Milk": 9656, "Grocery": 7561, "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338}'
```

**Dockerfile:** incluye un `HEALTHCHECK` que consulta `/health`
automáticamente cada 30s. Las dependencias están fijadas con versiones
exactas (no rangos) para garantizar reproduciibilidad — evita el
problema "en mi computadora sí funciona" por incompatibilidad de
versiones de scikit-learn al deserializar el modelo con joblib.

**`.dockerignore`:** excluye `venv/`, `data/`, `mlflow.db`, `mlruns/`,
`mlartifacts/`, `notebooks/`, `.git/` — nada de esto entra a la imagen.

## 12. API de inferencia

API REST construida con **FastAPI** que sirve predicciones de clustering
(sección M del enunciado). Carga el modelo K-Means y el FeatureBuilder
desde archivos joblib exportados por `train.py` — **no depende de
MLflow ni de la base de datos** en tiempo de inferencia.

**Código fuente:** `src/api/main.py` + `src/api/schemas.py`

**Levantar localmente (sin Docker):**

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Documentación interactiva (Swagger):** `http://localhost:8000/docs`

**Probar la API (PowerShell):**

```powershell
# Verificar salud
Invoke-RestMethod -Uri http://localhost:8000/health

# Predecir cluster para un cliente
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"Fresh": 12669, "Milk": 9656, "Grocery": 7561, "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338}'
```

### Endpoints

| Método | Ruta       | Descripción                                      |
| ------ | ---------- | ------------------------------------------------ |
| `GET`  | `/`        | Información básica de la API                     |
| `GET`  | `/health`  | Estado del servicio y versión del modelo cargado |
| `POST` | `/predict` | Predicción de cluster para un cliente nuevo      |

### Formato de request (POST /predict)

El endpoint acepta las 6 categorías de gasto del dataset original.
`Channel` y `Region` **no se piden** a propósito — el modelo fue
entrenado sin usarlas como input.

```json
{
  "Fresh": 12669,
  "Milk": 9656,
  "Grocery": 7561,
  "Frozen": 214,
  "Detergents_Paper": 2674,
  "Delicassen": 1338
}
```

### Formato de respuesta

```json
{
  "cluster": 1,
  "distance_to_centroid": 2.1244,
  "model_version": "5"
}
```

Los campos `Channel` y `Region` se completan internamente como `NaN`
para satisfacer la forma esperada por `FeatureBuilder` — no afectan el
cálculo del cluster ni la distancia.

### Artefactos cargados al iniciar

| Archivo                           | Origen                                 | Propósito                                 |
| --------------------------------- | -------------------------------------- | ----------------------------------------- |
| `models/kmeans_production.joblib` | `train.py` (en promoción a Production) | Modelo K-Means entrenado                  |
| `models/feature_builder.joblib`   | `train.py` (en promoción a Production) | Transformaciones ajustadas (scaler + PCA) |
| `models/production_metadata.json` | `train.py` (en promoción a Production) | Versión, métricas, run_id de MLflow       |

**Error si los artefactos no existen:** la API levanta pero retorna
HTTP 503 en `/predict` hasta que se ejecuten `python src/training/train.py`
para generar los modelos.

**Ejecutar dentro de Docker:** ver sección 11 (Docker).

## 13. Monitoring

_(pendiente — próxima entrega)_

## 14. Results

_(pendiente — próxima entrega)_

## 15. Team

| Integrantes           |
| --------------------- |
| Ismael Quesada Salas  |
| Magaly Bushey Ventura |
