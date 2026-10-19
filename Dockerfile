# One image, two roles (API and UI) - see docker-compose.yml.
# Building needs internet once (packages + search models); the running containers don't.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller; the heavy lifting (the language model) runs in Ollama.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
# Bake the search models into the image so it never downloads anything at runtime.
RUN python scripts/download_models.py

COPY ui ./ui
COPY .streamlit ./.streamlit
COPY evaluation ./evaluation

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
