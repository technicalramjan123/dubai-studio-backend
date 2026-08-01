FROM python:3.11-slim

# ffmpeg is required for audio/video extraction, chunking, and merging.
# build-essential/rustc/cargo are required to compile some of
# openai-whisper's sub-dependencies (e.g. tiktoken) which don't ship
# pre-built wheels for every platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl build-essential rustc cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STORAGE_DIR=/app/storage
ENV WHISPER_MODEL_SIZE=base
RUN mkdir -p /app/storage

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
