"""Punto de entrada de la aplicación Streamlit: Film Junky Union MVP."""
import streamlit as st

from film_junky_union.app.views.descriptive import render_descriptive_view
from film_junky_union.app.views.moderation import render_moderation_view
from film_junky_union.config import MODEL_PATH, PROCESSED_CATALOG_PATH


def main():
    st.set_page_config(
        page_title="Film Junky Union — MVP Analítico",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🎬 Film Junky Union")
    st.sidebar.markdown(
        "**Comunidad de Cine Clásico**  \n"
        "Plataforma inteligente de diagnóstico de catálogo y moderación asistida de reseñas."
    )
    st.sidebar.markdown("---")

    # Selector de módulos
    opcion = st.sidebar.radio(
        "Navegación del MVP:",
        [
            "📊 Módulo 1: Diagnóstico y Auditoría",
            "🛡️ Módulo 2: Asistente de Moderación",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Estado de Artefactos")
    datos_ok = PROCESSED_CATALOG_PATH.exists()
    modelo_ok = MODEL_PATH.exists()

    st.sidebar.write(f"• Datos procesados: {'✅ Listo' if datos_ok else '❌ Pendiente'}")
    st.sidebar.write(f"• Modelo entrenado: {'✅ Listo' if modelo_ok else '❌ Pendiente'}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Film Junky Union MVP — Versión 0.1.0 (Etapa 2)")

    if opcion == "📊 Módulo 1: Diagnóstico y Auditoría":
        render_descriptive_view()
    else:
        render_moderation_view()


if __name__ == "__main__":
    main()

