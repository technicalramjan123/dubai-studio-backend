FROM python:3.11-slim

# ffmpeg is required for audio/video extraction, chunking, and merging.
# git is required by openai-whisper's setup in some environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl \
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
