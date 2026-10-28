FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv para instalación rápida y reproducible
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias
COPY pyproject.toml ./

# Instalar dependencias del proyecto en el entorno del sistema
RUN uv pip install --system -r pyproject.toml

# Copiar código fuente, configuración y artefactos requeridos
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY .streamlit/ /app/.streamlit/

# Establecer PYTHONPATH
ENV PYTHONPATH="/app/src"

# Puerto predeterminado de Streamlit
EXPOSE 8501

# Configuración de salud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicio de la aplicación
ENTRYPOINT ["streamlit", "run", "src/film_junky_union/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
