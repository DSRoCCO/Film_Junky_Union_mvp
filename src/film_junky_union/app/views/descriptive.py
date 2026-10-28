"""Módulo 1: Diagnóstico y Auditoría del Catálogo de la Comunidad."""
import pandas as pd
import streamlit as st

from film_junky_union.config import (
    PROCESSED_CATALOG_PATH,
    PROCESSED_GENRE_DECADE_PATH,
)


@st.cache_data
def load_analytical_data():
    """Carga los artefactos analíticos procesados."""
    catalog = None
    genre_decade = None
    if PROCESSED_CATALOG_PATH.exists():
        catalog = pd.read_csv(PROCESSED_CATALOG_PATH)
    if PROCESSED_GENRE_DECADE_PATH.exists():
        genre_decade = pd.read_csv(PROCESSED_GENRE_DECADE_PATH)
    return catalog, genre_decade


def render_descriptive_view():
    """Renderiza el Módulo 1: Diagnóstico y Auditoría."""
    st.header("Módulo 1: Diagnóstico y Auditoría del Catálogo")
    st.caption(
        "Lectura ejecutiva del estado, calidad, cobertura y recepción crítica en la comunidad Film Junky Union."
    )

    catalog, genre_decade = load_analytical_data()

    if catalog is None or genre_decade is None:
        st.warning(
            "Los artefactos procesados no se han generado aún. Ejecuta el pipeline ETL para cargarlos."
        )
        return

    # 1. Métricas Principales de Cobertura
    st.subheader("1. Cobertura y Dimensiones de la Fuente")
    col1, col2, col3, col4 = st.columns(4)
    total_reseñas = catalog["reseñas"].sum()
    total_peliculas = len(catalog)
    tasa_global_neg = (catalog["negativas"].sum() / total_reseñas) * 100

    col1.metric("Total Reseñas", f"{total_reseñas:,}")
    col2.metric("Películas Únicas", f"{total_peliculas:,}")
    col3.metric("Rango Histórico", "1894 - 2010")
    col4.metric("Tasa Negativa Global", f"{tasa_global_neg:.1f}%")

    st.markdown("---")

    # 2. Auditoría de Calidad y Gobernanza
    st.subheader("2. Auditoría de Calidad y Gobernanza de Datos")
    st.markdown(
        """
    - **Esquema validado**: 47,331 reseñas sobre 6,648 obras.
    - **Nulos estructurales**: `end_year` (95.2% ausente, esperado en largometrajes) y `runtime_minutes` (1.0% ausente).
    - **Mitigación de duplicados y fugas**: Se detectaron 91 duplicados textuales. La partición oficial `ds_part` agrupa las películas por `tconst`, garantizando cero fuga de información entre train y test.
    - **Balance de clases**: Prácticamente perfecto (50.1% críticas negativas vs 49.9% no negativas), derivado de la exclusión en origen de puntuaciones intermedias (5 y 6).
    """
    )

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        st.write("**Top 10 Géneros por Volumen y Tasa de Negatividad**")
        genre_summary = (
            genre_decade.groupby("género")
            .agg(reseñas=("reseñas", "sum"), negativas=("negativas", "sum"))
            .assign(tasa_negativa=lambda x: (x["negativas"] / x["reseñas"]).mul(100).round(2))
            .sort_values("reseñas", ascending=False)
            .head(10)
        )
        st.dataframe(
            genre_summary[["reseñas", "tasa_negativa"]],
            column_config={
                "reseñas": st.column_config.NumberColumn("Total Reseñas", format="%d"),
                "tasa_negativa": st.column_config.ProgressColumn(
                    "% Críticas Negativas", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
            width="stretch",
        )

    with col_q2:
        st.write("**Evolución de Tasa Negativa por Década**")
        decade_summary = (
            genre_decade.groupby("decada")
            .agg(reseñas=("reseñas", "sum"), negativas=("negativas", "sum"))
            .assign(tasa_negativa=lambda x: (x["negativas"] / x["reseñas"]).mul(100).round(2))
            .query("decada >= 1920")
            .sort_index()
        )
        st.dataframe(
            decade_summary[["reseñas", "tasa_negativa"]],
            column_config={
                "reseñas": st.column_config.NumberColumn("Reseñas", format="%d"),
                "tasa_negativa": st.column_config.NumberColumn("% Negativa", format="%.1f%%"),
            },
            width="stretch",
        )

    st.markdown("---")

    # 3. Evidencia y Recepción por Película (Intervalos de Wilson)
    st.subheader("3. Recepción Confiable por Título (Intervalos de Wilson al 95%)")
    st.info(
        "💡 **Principio metodológico**: Una película con 100% de críticas negativas en 2 reseñas posee alta incertidumbre estadística. "
        "El intervalo de Wilson y la tasa suavizada evitan confundir muestras pequeñas con consenso."
    )

    min_reviews = st.slider(
        "Filtrar películas con al menos N reseñas en la comunidad:",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    filtered_catalog = catalog[catalog["reseñas"] >= min_reviews].copy()

    tab_pos, tab_neg = st.tabs(
        ["🌟 Películas con Mayor Consenso Positivo", "⚠️ Películas con Mayor Consenso Negativo"]
    )

    with tab_pos:
        top_pos = filtered_catalog.sort_values(
            ["tasa_suavizada", "reseñas"], ascending=[True, False]
        ).head(10)
        st.dataframe(
            top_pos[
                [
                    "primary_title",
                    "start_year",
                    "average_rating",
                    "reseñas",
                    "tasa_negativa",
                    "tasa_suavizada",
                    "ci_lower",
                    "ci_upper",
                ]
            ],
            column_config={
                "primary_title": "Título",
                "start_year": "Año",
                "average_rating": "Rating IMDb",
                "reseñas": "N° Reseñas",
                "tasa_negativa": st.column_config.NumberColumn("% Negativa Observada", format="%.1f%%"),
                "tasa_suavizada": st.column_config.NumberColumn("% Suavizada Empírica", format="%.1f%%"),
                "ci_lower": st.column_config.NumberColumn("IC 95% Inferior", format="%.1f%%"),
                "ci_upper": st.column_config.NumberColumn("IC 95% Superior", format="%.1f%%"),
            },
            hide_index=True,
            width="stretch",
        )

    with tab_neg:
        top_neg = filtered_catalog.sort_values(
            ["tasa_suavizada", "reseñas"], ascending=[False, False]
        ).head(10)
        st.dataframe(
            top_neg[
                [
                    "primary_title",
                    "start_year",
                    "average_rating",
                    "reseñas",
                    "tasa_negativa",
                    "tasa_suavizada",
                    "ci_lower",
                    "ci_upper",
                ]
            ],
            column_config={
                "primary_title": "Título",
                "start_year": "Año",
                "average_rating": "Rating IMDb",
                "reseñas": "N° Reseñas",
                "tasa_negativa": st.column_config.NumberColumn("% Negativa Observada", format="%.1f%%"),
                "tasa_suavizada": st.column_config.NumberColumn("% Suavizada Empírica", format="%.1f%%"),
                "ci_lower": st.column_config.NumberColumn("IC 95% Inferior", format="%.1f%%"),
                "ci_upper": st.column_config.NumberColumn("IC 95% Superior", format="%.1f%%"),
            },
            hide_index=True,
            width="stretch",
        )

