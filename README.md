# DubAI Studio — Backend

Free/open-source AI dubbing backend: yt-dlp (link download) + ffmpeg
(extraction/chunking/merging) + Whisper (transcription) + LibreTranslate
(translation, with a free Google Translate fallback) + Edge-TTS (voice
generation, generic voices — no cloning).

Built for long files (up to 3 hours) via silence-aware chunked processing,
with per-chunk retry and resumable jobs.

## Local run

```bash
pip install -r requirements.txt
# needs ffmpeg and yt-dlp installed on the system as well:
#   apt-get install ffmpeg
#   pip install yt-dlp
uvicorn main:app --reload
```

## Environment variables (all optional, sensible defaults included)

| Variable | Default | Purpose |
|---|---|---|
| `STORAGE_DIR` | `./storage` | where uploads/results are kept |
| `WHISPER_MODEL_SIZE` | `base` | `tiny` for lower RAM usage, `small`/`medium` for better accuracy |
| `LIBRETRANSLATE_URL` | `https://libretranslate.com/translate` | point this at your own self-hosted instance if you have one |
| `LIBRETRANSLATE_API_KEY` | *(empty)* | only needed for some public instances |
| `MAX_CONCURRENT_JOBS` | `1` | how many dubbing jobs run in parallel — keep at 1 on free-tier hosting |

## Deploy on Railway

1. Push this folder to a GitHub repo.
2. On [railway.app](https://railway.app): **New Project → Deploy from GitHub repo** → select the repo.
3. Railway detects the `Dockerfile` automatically and builds it (ffmpeg gets installed inside the container).
4. Once deployed, Railway gives you a public URL like `https://your-app.up.railway.app`.
5. Point your frontend's API calls at that URL.

## API endpoints

- `POST /jobs/upload` — multipart form: `file`, `target_language`, `voice`
- `POST /jobs/link` — form: `url`, `target_language`, `voice`, `media_duration_minutes`
- `GET /jobs` — list job history
- `GET /jobs/{id}` — job status + per-chunk progress
- `GET /jobs/{id}/download` — final dubbed file
- `GET /jobs/{id}/srt` — subtitle export
- `DELETE /jobs/{id}` — remove a job and its files
- `GET /health` — checks ffmpeg/yt-dlp are available

## Notes on free-tier limits

- Whisper's `base` model needs roughly 1GB RAM. If Railway's free tier
  runs out of memory on long files, switch `WHISPER_MODEL_SIZE` to `tiny`.
- `MAX_CONCURRENT_JOBS` is set to 1 by default — running multiple large
  jobs at once will likely exceed free-tier CPU/RAM.
- A 2–3 hour file will take a long time to process on free-tier hardware.
  Test with a short (5–10 min) file first to confirm the pipeline works
  end to end before trying long files.
