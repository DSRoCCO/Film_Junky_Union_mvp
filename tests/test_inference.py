"""Pruebas unitarias para el motor de inferencia y reglas operativas."""
import pytest
from pathlib import Path

from film_junky_union.config import MODEL_PATH, METADATA_PATH, MIN_F1_THRESHOLD
from film_junky_union.models.inference import ReviewModerator, load_model_and_metadata


def test_model_and_metadata_exist_and_meet_contract():
    pipeline, metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
    assert pipeline is not None
    assert metadata is not None

    test_metrics = metadata.get("standard_test_metrics", {})
    f1_negative = test_metrics.get("f1_negative", 0.0)
    assert f1_negative >= MIN_F1_THRESHOLD, f"F1 negativa ({f1_negative}) no alcanza el umbral de {MIN_F1_THRESHOLD}"
    assert test_metrics.get("passed_contract") is True


def test_review_moderator_positive_review():
    moderator = ReviewModerator()
    positive_text = "This movie is an absolute masterpiece! Brilliant acting, incredible direction and great music."
    result = moderator.predict(positive_text)

    assert result["margen"] < 0.0, "Una reseña abiertamente positiva debe tener margen negativo."
    assert result["decision"] == "Publicación Automática"
    assert result["badge_color"] == "green"
    assert result["es_negativa_estandar"] is False


def test_review_moderator_negative_review():
    moderator = ReviewModerator()
    negative_text = "The worst movie I have ever seen. Completely awful, terrible script, unwatchable trash. Avoid it!"
    result = moderator.predict(negative_text)

    assert result["margen"] > 0.291, "Una crítica abiertamente destructiva debe superar el umbral de alta certeza."
    assert result["decision"] == "Filtro Automático (Negativa de Alta Certeza)"
    assert result["badge_color"] == "red"
    assert result["es_negativa_estandar"] is True


def test_review_moderator_explainability():
    moderator = ReviewModerator()
    text = "The acting was terrible and the plot was awful, but the cinematography was beautiful."
    result = moderator.predict(text)

    explanations = result["explicacion"]
    assert "terminos_negativos" in explanations
    assert "terminos_positivos" in explanations

    neg_tokens = [item["termino"] for item in explanations["terminos_negativos"]]
    pos_tokens = [item["termino"] for item in explanations["terminos_positivos"]]

    # 'terrible' o 'awful' deben aparecer en términos negativos
    assert any(w in neg_tokens for w in ["terrible", "awful"])
    # 'beautiful' debe aparecer en términos positivos
    assert any(w in pos_tokens for w in ["beautiful"])

