"""Configuración global y constantes del MVP Film Junky Union."""
from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Rutas de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "imdb_reviews.tsv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CATALOG_PATH = PROCESSED_DATA_DIR / "catalog_summary.csv"
PROCESSED_GENRE_DECADE_PATH = PROCESSED_DATA_DIR / "genre_decade_summary.csv"

# Rutas de modelos
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Columnas esperadas en el archivo crudo
EXPECTED_RAW_COLUMNS = {
    "tconst", "title_type", "primary_title", "original_title",
    "start_year", "end_year", "runtime_minutes", "is_adult",
    "genres", "average_rating", "votes", "review",
    "rating", "sp", "pos", "ds_part", "idx"
}

# Parámetros analíticos y de modelado
RANDOM_STATE = 42
TARGET_COL = "is_negative"
MIN_F1_THRESHOLD = 0.85
OPERATING_THRESHOLD = 0.291  # Umbral de margen para alta precisión (>=94.6% en test)
AMBIGUITY_MARGIN_RANGE = (-0.2, 0.291)  # Rango para revisión humana asistida

