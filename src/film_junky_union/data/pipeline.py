"""Pipeline de datos: validación, limpieza y construcción de artefactos analíticos."""
import re
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from film_junky_union.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_CATALOG_PATH,
    PROCESSED_GENRE_DECADE_PATH,
    EXPECTED_RAW_COLUMNS,
    TARGET_COL,
)


def clean_text(text: str) -> str:
    """Normaliza texto: espacios en blanco compactos, manejo de nulos."""
    if not isinstance(text, str):
        return ""
    # Compactar secuencias de espacios en blanco
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def compute_wilson_interval(
    successes: pd.Series, trials: pd.Series, confidence: float = 0.95
) -> Tuple[pd.Series, pd.Series]:
    """Calcula el intervalo de puntuación de Wilson al 95% para proporciones binomiales."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / trials
    denominator = 1 + (z**2) / trials
    centre = (p + (z**2) / (2 * trials)) / denominator
    spread = (z / denominator) * np.sqrt(
        (p * (1 - p) / trials) + (z**2) / (4 * (trials**2))
    )
    lower = np.clip(centre - spread, 0.0, 1.0)
    upper = np.clip(centre + spread, 0.0, 1.0)
    return lower, upper


def load_and_validate_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga los datos crudos, valida el esquema y crea la columna objetivo."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"El archivo de datos crudos no fue encontrado en: {filepath}"
        )

    df = pd.read_csv(filepath, sep="\t", na_values=r"\N", low_memory=False)

    # Validar presencia de columnas esperadas
    missing_cols = EXPECTED_RAW_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columnas ausentes en el archivo crudo: {missing_cols}")

    # Validar coherencia del target pos (0 o 1)
    if not set(df["pos"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Valores inesperados en la columna 'pos' (debe ser 0 o 1).")

    # Crear target operativo: 1 = crítica negativa, 0 = no negativa
    df[TARGET_COL] = 1 - df["pos"]

    # Validar partición oficial
    if not set(df["ds_part"].dropna().unique()).issubset({"train", "test"}):
        raise ValueError("La columna 'ds_part' debe contener exclusivamente 'train' y 'test'.")

    return df


def build_analytical_artifacts(
    df: pd.DataFrame, output_dir: Path = PROCESSED_DATA_DIR
) -> Dict[str, Path]:
    """Genera y guarda los artefactos analíticos consumidos por la aplicación Streamlit."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resumen por película con intervalos de Wilson y tasa suavizada
    movie_agg = (
        df.groupby(["tconst", "primary_title"], dropna=False)
        .agg(
            reseñas=(TARGET_COL, "size"),
            negativas=(TARGET_COL, "sum"),
            start_year=("start_year", "first"),
            average_rating=("average_rating", "first"),
            votes=("votes", "first"),
            genres=("genres", "first"),
        )
        .reset_index()
    )

    movie_agg["tasa_negativa"] = (
        movie_agg["negativas"] / movie_agg["reseñas"]
    ).mul(100).round(2)

    ci_low, ci_high = compute_wilson_interval(
        movie_agg["negativas"], movie_agg["reseñas"]
    )
    movie_agg["ci_lower"] = ci_low.mul(100).round(2)
    movie_agg["ci_upper"] = ci_high.mul(100).round(2)

    # Suavizado bayesiano empírico hacia la media global
    global_mean_neg = df[TARGET_COL].mean()
    m_weight = 10.0  # Peso a priori equivalente a 10 reseñas
    movie_agg["tasa_suavizada"] = (
        (movie_agg["negativas"] + m_weight * global_mean_neg)
        / (movie_agg["reseñas"] + m_weight)
    ).mul(100).round(2)

    # Guardar catálogo
    movie_agg.to_csv(PROCESSED_CATALOG_PATH, index=False)

    # 2. Resumen por género y década
    genre_exploded = df.assign(
        género=df["genres"].fillna("Sin género").str.split(","),
        decada=(df["start_year"] // 10 * 10).astype("Int64"),
    ).explode("género")

    genre_decade = (
        genre_exploded.groupby(["género", "decada"], observed=True)
        .agg(
            reseñas=(TARGET_COL, "size"),
            negativas=(TARGET_COL, "sum"),
        )
        .reset_index()
    )
    genre_decade["tasa_negativa"] = (
        genre_decade["negativas"] / genre_decade["reseñas"]
    ).mul(100).round(2)

    genre_decade.to_csv(PROCESSED_GENRE_DECADE_PATH, index=False)

    return {
        "catalog_summary": PROCESSED_CATALOG_PATH,
        "genre_decade_summary": PROCESSED_GENRE_DECADE_PATH,
    }


def prepare_model_splits(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepara las particiones de entrenamiento y prueba garantizando cero fuga."""
    train_raw = df[df["ds_part"] == "train"].copy()
    test_raw = df[df["ds_part"] == "test"].copy()

    # Excluir duplicados internos de texto en train
    train_clean = train_raw.drop_duplicates(subset=["review"]).copy()

    # Limpiar textos
    X_train = train_clean["review"].apply(clean_text)
    y_train = train_clean[TARGET_COL].astype(int)

    X_test = test_raw["review"].apply(clean_text)
    y_test = test_raw[TARGET_COL].astype(int)

    # Verificación de integridad y no-fuga
    identical_overlap = X_train.isin(X_test)
    if identical_overlap.any():
        # Excluir de train textos que coincidan exactamente con test
        X_train = X_train[~identical_overlap]
        y_train = y_train[~identical_overlap]

    return X_train, y_train, X_test, y_test

