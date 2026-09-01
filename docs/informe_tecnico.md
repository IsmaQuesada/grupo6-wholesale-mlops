# Informe Técnico — Segmentación de Clientes Mayoristas

**Proyecto:** MLOps End-to-End: Del dato crudo a un sistema de ML en producción
**Grupo:** 6 — Clustering (Wholesale Customers)
**Integrantes:** Ismael Quesada Salas, Magaly Bushey Ventura
**Fecha:** Agosto 2026

---

## 1. Resumen Ejecutivo

Se implementó un sistema completo de MLOps para segmentar clientes de un distribuidor mayorista usando K-Means (k=3) sobre 6 categorías de gasto. El pipeline cubre ingesta reproducible, calidad de datos automatizada, ingeniería de variables reutilizable, experiment tracking con MLflow, serving via Docker + FastAPI, y monitoreo multidimensional (sistema, datos, modelo). Se identificaron3 perfiles de negocio: HoReCa Fresh (30.7%), Retail/Abarrotes (37.7%) y HoReCa Diversificado (31.6%), con silhouette de 0.257 y estabilidad verificada (diff=0.0002 entre semillas).

---

## 2. Arquitectura MLOps

El sistema se compone de9 capas:

```
UCI Repository → Ingestion → Data Quality → Feature Engineering
    → Training → MLflow → Model Registry → Docker/API → Monitoring
```

Cada capa es independiente y testeable. Los artefactos se serializan con joblib y se versionan en MLflow. El API no depende de MLflow en runtime — carga directamente los `.joblib` de `models/`.

**Diagrama completo:** Ver `docs/architecture.mmd` o `docs/architecture.png`.

---

## 3. Ingesta y Calidad de Datos

### 3.1 Estrategia de Ingesta

El script `src/ingestion/ingest.py` implementa una estrategia de3 tiers:

1. **ucimlrepo** — Paquete oficial de UCI (preferido)
2. **URL directa** — CSV desde el repositorio UCI (fallback)
3. **Copia local** — `data/raw/_fallback/` (offline/demo)

Cada ejecución genera `wholesale_customers_raw.csv` + `ingestion_metadata.json` con SHA-256 checksum y timestamp. La ingesta es idempotente.

### 3.2 Data Quality Gates

`src/data_quality/validate.py` implementa7 validaciones automáticas:

| Regla | Qué valida | Umbral | Justificación |
|-------|-----------|--------|---------------|
| `min_rows` | No haya pérdida significativa de filas | ≥90% de 440 | Dataset tiene 440 filas; 10% de pérdida máxima tolerable |
| `sin_nulos_obligatorios` | Sin nulos en variables de gasto | 0 nulos | Variables de gasto son el input del modelo |
| `duplicados_bajo_umbral` | Proporción de duplicados | <2% | Duplicados inflan métricas de clustering |
| `tipos_numericos_validos` | Variables de gasto sean numéricas | 100% numéricas | KMeans requiere input numérico |
| `sin_gastos_negativos` | Sin valores negativos | 0 negativos | Gasto no puede ser negativo |
| `cardinalidad_categórica_valida` | Channel/Region en valores esperados | Sin categorías nuevas | Evita datos corruptos |
| `esquema_sin_columnas_extra` | Solo8 columnas esperadas | 0 extra | Evita leak o error de esquema |

Las fallas se registran en `logs/quality_alerts.log` con timestamp para auditoría.

---

## 4. Ingeniería de Variables

### 4.1 FeatureBuilder

`src/features/build_features.py` define la clase `FeatureBuilder`, que implementa la interfaz scikit-learn (`fit_transform`/`transform`). Es la **única fuente de verdad** para transformaciones — no se reimplementa lógica en notebooks ni en producción.

### 4.2 Transformaciones Aplicadas

| # | Transformación | Justificación |
|---|---------------|---------------|
| 1 | `log1p` en 6 variables de gasto | Corrige alta asimetría (skewness) presente en Fresh, Milk, Grocery, etc. |
| 2 | Proporciones por categoría | Normaliza por volumen total de compra — permite comparar perfiles independientemente del tamaño |
| 3 | Ratio perecible/no-perecible | Separador de tipo de negocio: restaurantes (alto perecible) vs. tiendas (alto no-perecible) |
| 4 | Índice de diversificación (Shannon) | Mide qué tan distribuido está el gasto — clientes diversificados vs. especializados |
| 5 | StandardScaler | Estandariza para que PCA y KMeans no se sesguen por escala |
| 6 | PCA (5 componentes) | Reduce dimensionalidad manteniendo 85.7% de varianza; elimina correlación entre variables |

### 4.3 Selección de Componentes PCA

| Componente | Varianza | Acumulada |
|------------|----------|-----------|
| PC1 | 41.8% | 41.8% |
| PC2 | 16.3% | 58.2% |
| PC3 | 12.1% | 70.3% |
| PC4 | 9.2% | 79.5% |
| PC5 | 6.2% | **85.7%** |

Se seleccionaron5 componentes porque es el número mínimo que supera el umbral de 80% de varianza acumulada.

---

## 5. Modelado y Experimentación

### 5.1 Comparación de Configuraciones

Se evaluaron10 configuraciones (KMeans y Agglomerative Clustering, k=2 a k=6):

| Algoritmo | k | Silhouette | Davies-Bouldin | Inertia |
|-----------|---|------------|----------------|---------|
| KMeans | 2 | 0.3224 | 1.2315 | 3410.1 |
| **KMeans** | **3** | **0.2566** | **1.4325** | **2841.5** |
| KMeans | 4 | 0.2320 | 1.4155 | 2474.2 |
| KMeans | 5 | 0.2344 | 1.3935 | 2220.4 |
| KMeans | 6 | 0.2334 | 1.3506 | 2016.1 |
| Agglomerative | 2 | 0.2916 | 1.2679 | — |
| Agglomerative | 3 | 0.2297 | 1.5110 | — |
| Agglomerative | 4 | 0.2259 | 1.4337 | — |
| Agglomerative | 5 | 0.1796 | 1.5087 | — |
| Agglomerative | 6 | 0.1715 | 1.4550 | — |

KMeans domina a Agglomerative Clustering en todas las métricas para cada valor de k.

### 5.2 Justificación de k=3

Aunque k=2 tenía el mejor silhouette (0.322), su cluster dominante era 96% Channel=Horeca — básicamente redescubría una variable conocida del dataset. k=3 fue seleccionado porque:

- **Silhouette de 0.257** (segundo mejor, caída moderada desde 0.322)
- **Revela un perfil nuevo** (HoreCa diversificado) que Channel no captura
- **Sizes balanceados:** 30.7% / 37.7% / 31.6%
- **Estabilidad verificada:** diferencia de 0.0002 entre semillas (umbral: 0.05)

### 5.3 Perfiles de Clusters

| Cluster | Nombre | Gasto Promedio (Fresh / Milk / Grocery / Frozen / Detergents / Delicassen) | % |
|---------|--------|-----------------------------------------------------------------------------|---|
| 0 | HoReCa Fresh | 19,515 / 2,057 / 2,632 / 3,030 / 420 / 763 | 30.7% |
| 1 | Retail/Abarrotes | 6,221 / 9,596 / 14,870 / 1,266 / 6,550 / 1,283 | 37.7% |
| 2 | HoReCa Diversificado | 11,604 / 4,890 / 4,855 / 5,269 / 891 / 2,554 | 31.6% |

**Validación externa:** Los clusters 0 y 2 son 97% y 88% Horeca respectivamente, confirmando que representan dos comportamientos de compra genuinamente distintos dentro del mismo canal.

---

## 6. Experiment Tracking y Model Registry

### 6.1 MLflow

- **Backend:** SQLite (`mlflow.db` en raíz del repo)
- **Experimento:** `wholesale-clustering-grupo6`
- **Runs registrados:**2 (ambos con métricas idénticas, consistencia verificada)

**Parámetros por run:**
| Parámetro | Valor |
|-----------|-------|
| algorithm | KMeans |
| n_clusters | 3 |
| feature_set | PCA_5_componentes |
| random_seed | 42 |
| data_version | 20260901_043335 |

**Métricas por run:**
| Métrica | Valor |
|---------|-------|
| silhouette | 0.2566 |
| davies_bouldin | 1.4325 |
| inertia | 2841.48 |
| silhouette_semilla_alternativa | 0.2564 |
| diferencia_estabilidad | 0.0002 |

**Artefactos:** modelo (.joblib), FeatureBuilder (.joblib), cluster scatter plot (.png)

### 6.2 Model Registry

Ciclo implementado: Experiment → Candidate → Validation → Production

- **Nombre del modelo:** `wholesale-clustering-grupo6`
- **Versión en producción:** v2 (alias: `production`)
- **Ambas versiones:** Status `READY`

---

## 7. Serving y Despliegue

### 7.1 Docker

- **Base image:** `python:3.11-slim` (~150 MB)
- **Dependencias:** 7 paquetes (pinned a versiones exactas)
- **COPY selectivo:** Solo `src/api/`, `src/features/`, `src/monitoring/`, `models/` — sin código de entrenamiento, notebooks ni datos raw
- **Health check:** Python urllib cada30s contra `/health`
- **Puerto:** 8000

### 7.2 API (FastAPI)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info de la API |
| `/health` | GET | Estado del servicio + modelo cargado |
| `/metrics` | GET | Métricas del sistema (latency, throughput, errors) |
| `/predict` | POST | Predicción de cluster |

**Request:**
```json
{"Fresh": 12669, "Milk": 9656, "Grocery": 7561, "Frozen": 214, "Detergents_Paper": 2674, "Delicassen": 1338}
```

**Response:**
```json
{"cluster": 1, "distance_to_centroid": 2.1244, "model_version": "2"}
```

**Validación:** Pydantic schemas con `ge=0` en todos los campos. HTTP 422 para input inválido, HTTP 503 si modelos no cargados.

---

## 8. Monitoreo

### 8.1 O1 — System Monitoring

Métricas en memoria con sliding window (1000 requests):
- **Latency:** Promedio de tiempo de respuesta
- **Throughput:** Requests por segundo
- **Error Rate:** % de requests con error
- **Availability:** % de tiempo disponible

### 8.2 O2 — Data Drift (PSI)

Population Stability Index con Laplace smoothing en las6 variables de gasto:

| Umbral | Clasificación | Acción |
|--------|--------------|--------|
| PSI < 0.10 | OK | Sin acción |
| 0.10 ≤ PSI < 0.25 | WARNING | Vigilar |
| PSI ≥ 0.25 | ALERT | Evaluar reentrenamiento |

**Justificación de umbrales:** Basados en la práctica estándar de la industria (https://www.oreilly.com/library/view/mastering-python/9781789953299/). Los umbrales no son leyes universales — se adaptan al contexto del negocio.

### 8.3 O3 — Model Monitoring

Monitoreo de distribución de clusters: si algún cluster cambia más de10pp entre batches, se considera inestable.

### 8.4 R — Retrain Trigger

Combina3 señales (no solo una):

1. **Drift de datos:** PSI máximo > 0.25
2. **Degradación de calidad:** Silhouette actual < 70% del baseline
3. **Inestabilidad de composición:** Diferencia máxima > 10pp

**¿Por qué drift solo NO implica reentrenar?** La simulación mostró que BATCH 2 (PSI=0.131, WARNING) mantuvo el modelo estable — el drift leve no alcanzó a mover clientes entre clusters de forma significativa. Reentrenar solo por drift desperdiciaría cómputo y arriesgaría inestabilidad.

---

## 9. Simulación de Producción

### 9.1 Simulación de Drift

Se dividieron los datos conceptualmente en:

| Batch | Descripción | PSI Máximo | Decisión |
|-------|-------------|------------|----------|
| REFERENCE | Datos originales (440 filas) | — | — |
| BATCH 1 | Sin drift (bootstrap) | 0.042 | OK |
| BATCH 2 | Drift leve (+20% Fresh/Milk, ruido alto) | 0.131 | OK |
| BATCH 3 | Drift fuerte (+120% Fresh/Frozen/Delicassen, ruido bajo) | 0.571 | REENTRENAR |

**Hallazgo clave:** BATCH 2 tiene drift real pero el modelo se mantiene estable — el sistema correctamente no reentrena solo porque cambió la distribución de entrada. BATCH 3 dispara reentrenamiento por inestabilidad de composición (15.73pp vs umbral de 10pp).

### 9.2 Simulación de Calidad

Se inyectaron7 tipos de contaminación sobre un batch:
- Missing values
- Duplicados
- Outliers extremos
- Tipos incorrectos
- Categorías desconocidas
- Modificación de esquema
- Valores negativos

El sistema de validación detectó, bloqueó y registró cada incidente.

---

## 10. Testing

Suite de51 tests (46 incondicionales + 5 condicionales):

| Archivo | Tests | Cobertura |
|---------|-------|-----------|
| `test_ingestion.py` | 4 | validate_schema, compute_checksum |
| `test_data.py` | 11 | schema, types, ranges, missing, duplicates, DQ gates |
| `test_model.py` | 8 | carga, predicción, FeatureBuilder, transform, roundtrip |
| `test_api.py` | 11 | happy path, missing fields, negative, string, null, empty, wrong |
| `test_monitoring.py` | 17 | PSI, distributions, retrain trigger, system metrics, boundaries |

---

## 11. Conclusiones

### Logros
- Pipeline completo de MLOps: ingesta → calidad → features → training → serving → monitoreo
- 51 tests automatizados
- 3 perfiles de negocio identificados con validación externa
- Sistema de monitoreo multidimensional con lógica de reentrenamiento justificada
- Docker + API funcionales y documentados

### Limitaciones
- Dataset pequeño (440 filas) — resultados en datasets mayores podrían variar
- Silhouette moderado (0.257) — inherente a la naturaleza del problema (solapamiento entre perfiles)
- Monitoreo en memoria — para producción se recomienda persistencia (Redis, time-series DB)

### Trabajos Futuros
- Probar algoritmos de clustering más avanzados (DBSCAN, Gaussian Mixture)
- Implementar reentrenamiento automático con validación humana
- Agregar monitoreo de outliers y anomalías en tiempo real
- Integrar con sistema de alertas externo (Slack, email)
