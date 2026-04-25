# Start directly from your cached 11.42 GB PyTorch image!
FROM pytorch/pytorch:latest

# ---- system deps ----
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- app code ----
WORKDIR /app
COPY model_service.py /app/model_service.py

# ---- runtime defaults ----
ENV NODE_NAME="edge-node" \
    INFERENCE_PORT="5000" \
    PYTHONUNBUFFERED="1"

EXPOSE 5000

HEALTHCHECK --interval=3s --timeout=2s --start-period=5s --retries=5 \
    CMD curl -sf http://localhost:5000/health || exit 1

CMD ["python", "model_service.py"]