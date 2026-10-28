"""Motor de inferencia, calibración y explicabilidad para moderación de reseñas."""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np

from film_junky_union.config import (
    MODEL_PATH,
    METADATA_PATH,
    OPERATING_THRESHOLD,
    AMBIGUITY_MARGIN_RANGE,
)
from film_junky_union.data.pipeline import clean_text


def load_model_and_metadata(
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> Tuple[Any, Dict[str, Any]]:
    """Carga el pipeline entrenado y sus metadatos asociados."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"El modelo no existe en {model_path}. Ejecuta primero el pipeline de entrenamiento."
        )
    pipeline = joblib.load(model_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata


class ReviewModerator:
    """Clase principal de inferencia y asistencia a la moderación."""

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.pipeline, self.metadata = load_model_and_metadata(model_path, metadata_path)
        self.vectorizer = self.pipeline.named_steps["vectorizer"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.coefficients = self.classifier.coef_[0]
        self.intercept = float(self.classifier.intercept_[0])
        self.default_threshold = self.metadata.get("operating_rule", {}).get(
            "operating_threshold", OPERATING_THRESHOLD
        )

    def explain_prediction(self, text: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Identifica los términos presentes en la reseña que más influyen en la decisión."""
        cleaned = clean_text(text)
        if not cleaned:
            return {"terminos_negativos": [], "terminos_positivos": []}

        tfidf_vec = self.vectorizer.transform([cleaned])
        nonzero_indices = tfidf_vec.indices
        tfidf_values = tfidf_vec.data

        if len(nonzero_indices) == 0:
            return {"terminos_negativos": [], "terminos_positivos": []}

        # Contribución local = tfidf_value * coefficient
        contributions = tfidf_values * self.coefficients[nonzero_indices]
        token_names = self.feature_names[nonzero_indices]

        # Separar términos que empujan a negativo (coef > 0) y positivo (coef < 0)
        neg_items = []
        pos_items = []

        for token, contrib, tfidf_val in zip(token_names, contributions, tfidf_values):
            item = {"termino": str(token), "impacto": round(float(contrib), 4), "tfidf": round(float(tfidf_val), 3)}
            if contrib > 0:
                neg_items.append(item)
            elif contrib < 0:
                pos_items.append(item)

        # Ordenar por magnitud de impacto
        neg_items.sort(key=lambda x: x["impacto"], reverse=True)
        pos_items.sort(key=lambda x: x["impacto"])  # Más negativo = más pro-positivo

        return {
            "terminos_negativos": neg_items[:top_k],
            "terminos_positivos": pos_items[:top_k],
        }

    def predict(
        self,
        text: str,
        threshold: float = None,
        ambiguity_min: float = AMBIGUITY_MARGIN_RANGE[0],
    ) -> Dict[str, Any]:
        """Evalúa una reseña y genera la decisión operativa y su explicación."""
        op_threshold = self.default_threshold if threshold is None else threshold
        cleaned = clean_text(text)

        if not cleaned:
            return {
                "error": "Texto vacío o inválido.",
                "decision": "Entrada Inválida",
                "margen": 0.0,
            }

        margin = float(self.pipeline.decision_function([cleaned])[0])
        is_negative_standard = bool(margin >= 0.0)

        # Determinar política de moderación
        if margin >= op_threshold:
            decision = "Filtro Automático (Negativa de Alta Certeza)"
            accion = "Bloquear o marcar para descarte directo"
            nivel_riesgo = "Alto (Certeza de negatividad >= 94%)"
            badge_color = "red"
        elif ambiguity_min <= margin < op_threshold:
            decision = "Revisión Humana Asistida"
            accion = "Enviar al panel de moderación con términos resaltados"
            nivel_riesgo = "Medio (Zona de ambigüedad / crítica mixta)"
            badge_color = "orange"
        else:
            decision = "Publicación Automática"
            accion = "Aprobar y publicar en la comunidad"
            nivel_riesgo = "Bajo (Crítica favorable / no negativa)"
            badge_color = "green"

        explanations = self.explain_prediction(cleaned)

        return {
            "texto_normalizado": cleaned,
            "palabras": len(cleaned.split()),
            "margen": round(margin, 4),
            "umbral_operativo_usado": round(op_threshold, 4),
            "es_negativa_estandar": is_negative_standard,
            "decision": decision,
            "accion_recomendada": accion,
            "nivel_riesgo": nivel_riesgo,
            "badge_color": badge_color,
            "explicacion": explanations,
        }

