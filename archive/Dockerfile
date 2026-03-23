FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    tiktoken \
    fastapi \
    uvicorn \
    datasets \
    huggingface_hub

COPY . /app/

EXPOSE 7860

CMD ["python", "chat.py", "--checkpoint", "checkpoints/best.pt", "--port", "7860"]
