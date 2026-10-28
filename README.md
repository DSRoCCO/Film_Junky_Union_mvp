# 🎬 Film Junky Union — Prototipo Analítico (MVP)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager: UV](https://img.shields.io/badge/UV-Package%20Manager-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![UI: Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ML: Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: Pytest 9.2](https://img.shields.io/badge/Tests-9%2F9%20Passed-44CC11?logo=pytest&logoColor=white)](https://pytest.org/)
[![Container: Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Sistema analítico modular y reproducible para el diagnóstico del catálogo de cine clásico y la moderación asistida de críticas de la comunidad **Film Junky Union**, basado en Procesamiento de Lenguaje Natural (NLP).

---

## 📌 1. Descripción del Proyecto y Caso de Negocio

**Film Junky Union** es una comunidad de aficionados al cine clásico en constante crecimiento. Con el incremento diario en el flujo de publicaciones, el equipo editorial enfrenta el reto de moderar críticas tóxicas, agresivas o excesivamente destructivas sin saturar la capacidad humana de revisión ni censurar opiniones constructivas por error.

El proyecto se estructuró en dos etapas metodológicas respaldadas por la evidencia analítica disponible en el cuaderno ejecutable [`notebooks/analisis_integral.ipynb`](notebooks/analisis_integral.ipynb) y su informe ejecutivo consolidado en PDF [`report/analisis_integral.pdf`](report/analisis_integral.pdf):

1. **Etapa 1: Análisis Integral de Datos y Modelado sin Fuga (`notebooks/analisis_integral.ipynb`)**:
   - Auditoría exhaustiva de calidad (47,331 reseñas sobre 6,648 películas estrenadas entre 1894 y 2010).
   - Mitigación de fuga (*Zero Data Leakage*): partición oficial agrupada por obra cinematográfica (`tconst`), exclusión de proxies directos (`rating`, `sp`) y metadatos no disponibles al momento de moderar.
   - Comparativa de modelos NLP garantizando un $F_1 \ge 0.85$ en la clase negativa sobre test oficial.
   - Diagnóstico de incertidumbre en la recepción por película mediante **Intervalos de Confianza de Wilson al 95%** y tasas suavizadas empíricas.
2. **Etapa 2: Conversión a MVP Analítico Modular (`src/film_junky_union/`)**:
   - Extracción de la lógica analítica a un paquete Python desacoplado y comprobable con `pytest`.
   - Pipeline reproducible de datos que publica artefactos canónicos en `data/processed/`.
   - Serialización del pipeline entrenado en `models/`.
   - Interfaz web interactiva en **Streamlit** dividida en dos módulos (Diagnóstico y Moderador asistido).
   - Empaquetado y contenerización determinista con **UV** y **Docker**.

---

## 📊 2. Resultados del Modelo y Criterio Contractual

| Modelo Evaluado | Representación | $F_1$ Negativa (CV) | $F_1$ Negativa (Test) | Accuracy | Decisión |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline trivial** | Estrategia mayoritaria | 0.000 | 0.000 | 50.15% | Descartado |
| **Count + Logistic Regression** | Bolsa de palabras | 0.884 | 0.883 | 88.24% | Descartado |
| **TF-IDF (1,2) + Logistic Regression** | Unigramas + Bigramas | 0.885 | 0.884 | 88.35% | Descartado |
| **TF-IDF (1,2) + LinearSVC** | Unigramas + Bigramas (`sublinear_tf`) | **0.890** | **0.8898** | **88.91%** | **SELECCIONADO** |

### Métricas Oficiales en Test no Visto (23,535 reseñas):
- **$F_1$ Negativa**: **`0.8898`** (Supera el umbral contractual de $\ge 0.85$, **APROBADO**).
- **Exactitud Global**: **`88.91%`**.
- **Precisión Negativa**: **`88.65%`** | **Recall Negativo**: **`89.32%`**.

### Regla Operativa de Alta Certeza (Margen $\ge 0.291$):
- **Precisión Negativa**: **`94.59%`** (Solo 5 de cada 100 reseñas filtradas automáticamente son falsos positivos).
- **Cobertura**: **`41.21%`** del flujo total.
- **Zona de Ambigüedad (Margen entre -0.20 y 0.291)**: Se deriva automáticamente a **Revisión Humana Asistida** en la aplicación.

---

## 🗂️ 3. Arquitectura del Repositorio

```text
Film_Junky_Union_mvp/
├── project_spec.md               # Especificación funcional y contrato analítico a MVP
├── pyproject.toml                # Especificación de dependencias (PEP 518/621)
├── uv.lock                       # Lockfile criptográfico congelado (84 dependencias)
├── requirements.txt              # Exportación fijada con hashes SHA-256 para pip clásico
├── Dockerfile                    # Receta de contenerización inmutable para producción
├── .dockerignore                 # Exclusión de datos crudos, tests, notebooks y cachés
├── .gitignore                    # Exclusión estricta de datos locales y skills
├── README.md                     # Documentación principal del proyecto
│
├── notebooks/
│   └── analisis_integral.ipynb   # Cuaderno integral con la evidencia congelada de Etapa 1
│
├── report/
│   └── analisis_integral.pdf     # Informe ejecutivo consolidado en PDF (Etapa 1)
│
├── data/
│   ├── imdb_reviews.tsv          # Fuente original cruda (excluida de Git/Docker)
│   └── processed/                # Artefactos procesados canónicos consumidos por el MVP
│       ├── catalog_summary.csv   # Resumen de títulos con intervalos de Wilson (95%)
│       └── genre_decade_summary.csv # Negatividad agregada por género y década
│
├── models/
│   ├── sentiment_pipeline.joblib # Pipeline Scikit-Learn optimizado y serializado
│   └── metadata.json             # Ficha técnica, fechas, métricas y umbrales
│
├── src/film_junky_union/         # Paquete Python modular y desacoplado
│   ├── __init__.py
│   ├── config.py                 # Constantes de rutas, semillas y umbrales de decisión
│   ├── data/
│   │   ├── __init__.py
│   │   └── pipeline.py           # Ingesta, validación, limpieza y cálculo de Wilson
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py           # Entrenamiento formal, evaluación y exportación
│   │   └── inference.py          # Inferencia, niveles de riesgo y explicabilidad local
│   └── app/
│       ├── __init__.py
│       ├── main.py               # Aplicación principal Streamlit con navegación
│       └── views/
│           ├── __init__.py
│           ├── descriptive.py    # Módulo 1: Diagnóstico ejecutivo y auditoría
│           └── moderation.py     # Módulo 2: Asistente interactivo y simulador
│
└── tests/                        # Suite de pruebas automatizadas (pytest)
    ├── test_data_pipeline.py     # Pruebas de limpieza y límites del intervalo de Wilson
    └── test_inference.py         # Pruebas de contrato F1, inferencia y tokens
```

---

## 🚀 4. Instalación y Replicación Rápida

### Opción A: Con UV (Recomendada — Instalación en ~2 segundos)
```bash
# 1. Crear el entorno virtual
uv venv .venv --python /usr/bin/python3

# 2. Sincronizar de forma inmutable con el lockfile congelado
uv sync --frozen
```

### Opción B: Con Pip Estándar
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ 5. Ejecución del Pipeline y Pruebas Unitarias

### Regeneración de Artefactos y Reentrenamiento
La lógica analítica puede ejecutarse de forma independiente desde terminal sin necesidad de abrir cuadernos ni la interfaz:

```bash
PYTHONPATH=src .venv/bin/python -c "
from film_junky_union.data.pipeline import load_and_validate_raw_data, build_analytical_artifacts
from film_junky_union.models.training import train_and_export_model

df = load_and_validate_raw_data()
build_analytical_artifacts(df)
train_and_export_model(df)
"
```

### Ejecución de Pruebas Automatizadas
```bash
PYTHONPATH=src .venv/bin/pytest -v
```
> **Resultado**: 9/9 pruebas pasadas en ~3 segundos (validación de esquema, límites de Wilson, contrato $F_1$, inferencia en tres estados y explicabilidad de n-gramas).

---

## 🖥️ 6. Lanzamiento de la Aplicación Streamlit

Para iniciar la interfaz interactiva con ambos módulos analíticos:

```bash
PYTHONPATH=src .venv/bin/streamlit run src/film_junky_union/app/main.py
```
> Abrir en el navegador local: **[http://localhost:8501](http://localhost:8501)**

### Funcionalidades de la Aplicación:
1. **Módulo 1: Diagnóstico y Auditoría del Catálogo**:
   - Cobertura de películas (6,648 obras), volumen de reseñas (47,331) y rango temporal (1894–2010).
   - Auditoría de datos: control de nulos estructurales (`end_year`, `runtime_minutes`), exclusión de duplicados y garantía de cero fuga entre particiones.
   - Tasas de negatividad por género (*Horror* 70.9% vs *Animación* 28.5%) y evolución por década.
   - **Tablas de Recepción Confiable**: filtro interactivo por volumen de reseñas con cálculo dinámico de **Intervalos de Confianza de Wilson al 95%** y tasas suavizadas empíricas para distinguir consenso real de muestras pequeñas.
2. **Módulo 2: Asistente Inteligente de Moderación**:
   - Inferencia en tiempo real sobre texto libre en inglés o selección de ejemplos predefinidos.
   - **Triage Operativo en 3 Niveles**:
     - 🛑 **Filtro Automático Directo**: Margen $\ge 0.291$ (certeza $\ge 94.6\%$).
     - ⚠️ **Revisión Humana Asistida**: Margen en zona de ambigüedad (críticas mixtas, irónicas o complejas).
     - ✅ **Publicación Automática**: Críticas con margen claramente favorable ($< -0.20$).
   - **Explicabilidad Léxica Local**: tabla comparativa que desglosa los unigramas y bigramas presentes en el texto que más empujaron la decisión hacia negativo o positivo.
   - **Simulador de Capacidad Editorial**: permite al equipo ajustar el volumen diario esperado y las horas-hombre disponibles, alertando en tiempo real si el flujo de moderación excede la capacidad del equipo.
   - Botón de reinicio de estado para restaurar los valores base en cualquier momento.

---

## 🐳 7. Despliegue con Docker

El proyecto cuenta con un `Dockerfile` optimizado con caché de dependencias y healthcheck integrado:

```bash
# 1. Construir la imagen Docker
docker build -t film-junky-union-mvp .

# 2. Ejecutar el contenedor
docker run -p 8501:8501 film-junky-union-mvp
```
Acceder a la aplicación contenerizada en: `http://localhost:8501`.

---

## 🛡️ 8. Gobernanza y Límites Metodológicos (No-Go)

- **Prevención de Fuga de Información**: El clasificador utiliza únicamente el texto no estructurado (`review`). Las calificaciones numéricas de usuario (`rating`) y la polaridad pre-asignada (`sp`) están explícitamente excluidas del vectorizador y del modelo.
- **Ausencia de Atributos de Usuario y Temporales de Posteo**: La fuente no incluye identificadores de revisor ni fecha/hora de publicación de cada reseña. En consecuencia, **no se contemplan algoritmos de recomendación personalizada ni análisis de series temporales de opinión**.
- **Carácter Condicional de las Simulaciones**: El simulador de capacidad y las políticas de umbral representan cálculos condicionados por supuestos estadísticos observados en el test oficial; no constituyen garantías ni pruebas de causalidad en entornos con distribuciones drásticamente alteradas.
