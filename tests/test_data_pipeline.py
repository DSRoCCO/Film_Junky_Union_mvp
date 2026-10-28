"""Pruebas unitarias para el pipeline de datos y transformaciones analíticas."""
import numpy as np
import pandas as pd
import pytest

from film_junky_union.config import (
    PROCESSED_CATALOG_PATH,
    PROCESSED_GENRE_DECADE_PATH,
    RAW_DATA_PATH,
)
from film_junky_union.data.pipeline import clean_text, compute_wilson_interval


def test_clean_text_normalizes_whitespace():
    raw = "  This   is \n a \t movie with    bad   acting.  "
    expected = "This is a movie with bad acting."
    assert clean_text(raw) == expected


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_compute_wilson_interval_bounds():
    successes = pd.Series([0, 5, 10, 20])
    trials = pd.Series([10, 10, 10, 20])
    lower, upper = compute_wilson_interval(successes, trials, confidence=0.95)

    assert (lower >= 0.0).all()
    assert (upper <= 1.0).all()
    assert (lower <= upper).all()
    # Para 0/10, el límite superior debe ser mayor a 0 pero moderado
    assert upper.iloc[0] > 0.0
    # Para 10/10, el límite inferior debe ser menor a 1 pero alto
    assert lower.iloc[2] < 1.0


def test_processed_catalog_artifact_exists_and_valid():
    assert PROCESSED_CATALOG_PATH.exists(), "El artefacto catalog_summary.csv debe existir."
    df = pd.read_csv(PROCESSED_CATALOG_PATH)

    expected_cols = {
        "tconst", "primary_title", "start_year", "average_rating",
        "votes", "genres", "reseñas", "negativas", "tasa_negativa",
        "ci_lower", "ci_upper", "tasa_suavizada"
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 5000, "El catálogo debe contener miles de títulos."
    assert (df["tasa_negativa"] >= 0.0).all() and (df["tasa_negativa"] <= 100.0).all()
    assert (df["ci_lower"] <= df["ci_upper"]).all()


def test_processed_genre_decade_artifact_exists_and_valid():
    assert PROCESSED_GENRE_DECADE_PATH.exists(), "El artefacto genre_decade_summary.csv debe existir."
    df = pd.read_csv(PROCESSED_GENRE_DECADE_PATH)

    expected_cols = {"género", "decada", "reseñas", "negativas", "tasa_negativa"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 50

