"""Entrenamiento, evaluación y exportación del pipeline de moderación."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from film_junky_union.config import (
    MODEL_PATH,
    METADATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    OPERATING_THRESHOLD,
    MIN_F1_THRESHOLD,
)
from film_junky_union.data.pipeline import load_and_validate_raw_data, prepare_model_splits

logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Construye el pipeline aprobado en el análisis integral."""
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
        ]
    )


def train_and_export_model(
    df: pd.DataFrame = None, output_model_path: Path = MODEL_PATH
) -> Dict[str, Any]:
    """Entrena el pipeline sobre train, evalúa en test y exporta artefactos y metadatos."""
    if df is None:
        df = load_and_validate_raw_data()

    X_train, y_train, X_test, y_test = prepare_model_splits(df)

    logger.info(
        f"Iniciando entrenamiento con {len(X_train)} muestras de train y {len(X_test)} de test."
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluación estándar en test
    y_pred = pipeline.predict(X_test)
    test_f1 = float(f1_score(y_test, y_pred, pos_label=1))
    test_precision = float(precision_score(y_test, y_pred, pos_label=1))
    test_recall = float(recall_score(y_test, y_pred, pos_label=1))
    test_accuracy = float(accuracy_score(y_test, y_pred))

    if test_f1 < MIN_F1_THRESHOLD:
        raise ValueError(
            f"El modelo no superó el criterio mínimo contractual de F1 >= {MIN_F1_THRESHOLD}. Obtenido: {test_f1:.4f}"
        )

    # Evaluación de la regla operativa de alta precisión
    test_margins = pipeline.decision_function(X_test)
    test_operating_pred = (test_margins >= OPERATING_THRESHOLD).astype(int)
    op_coverage = float(test_operating_pred.mean())
    op_precision = float(precision_score(y_test, test_operating_pred, zero_division=0))
    op_recall = float(recall_score(y_test, test_operating_pred, zero_division=0))
    op_f1 = float(f1_score(y_test, test_operating_pred, zero_division=0))

    # Guardar modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_model_path)

    # Guardar metadatos
    metadata = {
        "model_name": "TF-IDF (1,2) + LinearSVC",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "standard_test_metrics": {
            "f1_negative": round(test_f1, 4),
            "precision_negative": round(test_precision, 4),
            "recall_negative": round(test_recall, 4),
            "accuracy": round(test_accuracy, 4),
            "passed_contract": test_f1 >= MIN_F1_THRESHOLD,
        },
        "operating_rule": {
            "operating_threshold": OPERATING_THRESHOLD,
            "coverage": round(op_coverage, 4),
            "precision_negative": round(op_precision, 4),
            "recall_negative": round(op_recall, 4),
            "f1_negative": round(op_f1, 4),
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Modelo exportado con éxito a {output_model_path}. F1 Test: {test_f1:.4f}"
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_export_model()

