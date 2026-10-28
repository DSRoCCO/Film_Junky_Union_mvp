# Especificación Técnica y Contrato de MVP: Film Junky Union

## 1. Contexto y Problema de Negocio
**Film Junky Union** es una comunidad de aficionados al cine clásico que recibe un volumen creciente de reseñas. Para mantener la calidad del contenido y prevenir la saturación del equipo editorial, se requiere un prototipo analítico (MVP) que combine:
1. **Exploración y diagnóstico del catálogo**: Comprensión objetiva de cómo se distribuyen las opiniones por película, género, década y discrepancia con la crítica masiva de IMDb.
2. **Moderación y categorización inteligente**: Detección automática y priorización de críticas negativas mediante procesamiento de lenguaje natural (NLP), reduciendo la carga de revisión manual sin censurar opiniones legítimas por error.

## 2. Decisión Operativa Respaldada
- **Unidad de decisión**: Una reseña individual no estructurada (`review`).
- **Momento de decisión**: Al momento en que el usuario envía la reseña, antes de su publicación o derivación a moderación.
- **Flujo de decisión en 3 niveles**:
  - **Filtro Automático de Alta Confianza**: Si el margen de decisión es $\ge 0.291$ (precisión test $\approx 94.6\%$), la reseña se clasifica como negativa directa.
  - **Revisión Humana Asistida**: Si el margen está en la zona de ambigüedad (alrededor de 0), se envía al panel editorial con los términos detonantes resaltados.
  - **Publicación Automática**: Si el margen es claramente negativo ($< -0.2$), se publica sin requerir intervención.
- **Límites de alcance y no-go**:
  - No se dispone de identificador de usuario ni timestamp de publicación: **no aplica recomendación personalizada, series de tiempo ni atribución causal**.

## 3. Contrato de Datos y Mitigación de Fuga
- **Fuente primaria**: `data/imdb_reviews.tsv` (47,331 registros).
- **Features de entrada al modelo**: Exclusivamente el texto de la reseña (`review`).
- **Variables excluidas del modelo**:
  - `rating`, `sp`: Excluidas por ser proxies directos del target (fuga evidente).
  - `average_rating`, `votes`, metadatos de película: Excluidos de la inferencia para asegurar que el modelo evalúe estrictamente el contenido del texto sin sesgarse por la reputación previa de la obra.
- **Partición oficial**: Respeto estricto de `ds_part` (`train` = 23,796, `test` = 23,535). Las películas están agrupadas por `tconst` sin solapamiento efectivo.

## 4. Métricas de Rendimiento y Criterios de Aceptación
- **Target**: `is_negative = 1 - pos` (1 = crítica negativa, 0 = no negativa).
- **Criterio de éxito**: $F_1 \ge 0.85$ en la clase negativa sobre el conjunto de test oficial.
- **Resultados del análisis integral**:
  - Baseline trivial (clase mayoritaria): $F_1 = 0.0$
  - Pipeline seleccionado: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` + `LinearSVC(C=1.0)`
  - Desempeño en test oficial (23,535 muestras no vistas):
    - **$F_1$ negativa**: $0.890$ (APROBADO, supera 0.85)
    - **Accuracy**: $0.889$
    - **Precision negativa**: $0.886$
    - **Recall negativa**: $0.893$
  - Regla de alta precisión (umbral margen $\ge 0.291$):
    - **Precision**: $0.946$
    - **Cobertura**: $41.2\%$

## 5. Artefactos que Cruzan Etapas
| Artefacto | Tipo | Esquema Canónico | Productor | Consumidor |
| :--- | :--- | :--- | :--- | :--- |
| `data/processed/catalog_summary.csv` | Datos analíticos | `tconst`, `primary_title`, `start_year`, `average_rating`, `votes`, `genres`, `reseñas`, `negativas`, `tasa_negativa`, `ci_lower`, `ci_upper`, `tasa_suavizada` | Pipeline ETL | Streamlit (Módulo 1) |
| `data/processed/genre_decade_summary.csv` | Datos analíticos | `género`, `decada`, `reseñas`, `tasa_negativa` | Pipeline ETL | Streamlit (Módulo 1) |
| `models/sentiment_pipeline.joblib` | Modelo | Pipeline Scikit-Learn (`TfidfVectorizer` + `LinearSVC`) | Pipeline Entrenamiento | Inferencia / Streamlit (Módulo 2) |
| `models/metadata.json` | Metadatos | Versión, fecha, métricas oficiales de test, umbral operativo (0.291) y baseline | Pipeline Entrenamiento | Streamlit (Módulo 2) y Tests |

## 6. Arquitectura de la Interfaz Streamlit
- **Módulo 1: Diagnóstico y Auditoría de la Comunidad** (Estático, ejecutivo):
  - Calidad de datos, cobertura de películas, nulos y duplicados.
  - Tasas de negatividad por género cinematográfico y por década.
  - Tablas de recepción confiable con intervalos de Wilson (95%) para evitar conclusiones apresuradas sobre muestras pequeñas.
- **Módulo 2: Asistente Interactivo de Moderación** (Reactivo, prescriptivo):
  - Inferencia en tiempo real sobre texto ingresado por el usuario.
  - Simulador de umbral de margen y capacidad operativa de moderación.
  - Descomposición léxica: identificación de las palabras y bigramas que mayor peso tuvieron en la decisión.
  - Botón de reinicio de escenario (`Restablecer valores por defecto`).

