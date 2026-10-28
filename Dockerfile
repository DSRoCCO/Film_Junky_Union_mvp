FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias mínimas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv para gestión determinista de dependencias
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias y lockfile congelado
COPY pyproject.toml uv.lock ./

# Sincronizar el entorno virtual de forma inmutable a partir de uv.lock
RUN uv sync --frozen --no-install-project

# Copiar código fuente, artefactos analíticos y configuración
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY .streamlit/ /app/.streamlit/

# Configurar variables de entorno: el venv de UV tiene prioridad en PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Exponer el puerto predeterminado de Streamlit
EXPOSE 8501

# Verificación de salud del contenedor
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de arranque del MVP
ENTRYPOINT ["streamlit", "run", "src/film_junky_union/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
