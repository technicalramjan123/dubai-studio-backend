import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from models import init_db, get_session, Job, Chunk
import worker
import shutil as _shutil

STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")
MAX_DURATION_MINUTES = 180  # 3 hours cap
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

app = FastAPI(title="DubAI Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend's domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    init_db()


@app.get("/health")
def health():
    tools_ok = {
        "ffmpeg": _shutil.which("ffmpeg") is not None,
        "yt-dlp": _shutil.which("yt-dlp") is not None,
    }
    return {"status": "ok", "tools": tools_ok}


def _media_type_from_filename(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "video"  # sensible default; ffprobe will reveal the truth downstream


@app.post("/jobs/upload")
def create_job_from_upload(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    voice: str = Form("male"),
):
    job_id = str(uuid.uuid4())
    wd = worker.job_dir(job_id)
    dest_path = os.path.join(wd, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session = get_session()
    job = Job(
        id=job_id,
        source_type="upload",
        original_filename=file.filename,
        media_type=_media_type_from_filename(file.filename),
        target_language=target_language,
        voice=voice,
        input_path=dest_path,
        status="pending",
        current_step_label="Queued",
    )
    session.add(job)
    session.commit()
    session.close()

    worker.submit_job(job_id)
    return {"job_id": job_id}


@app.post("/jobs/link")
def create_job_from_link(
    url: str = Form(...),
    target_language: str = Form(...),
    voice: str = Form("male"),
    media_duration_minutes: float = Form(None),
):
    if media_duration_minutes and media_duration_minutes > MAX_DURATION_MINUTES:
        raise HTTPException(
            status_code=400,
            detail=(f"This file is longer than the {MAX_DURATION_MINUTES}-minute cap. "
                    "Longer files take proportionally more time and free-tier resources.")
        )

    job_id = str(uuid.uuid4())
    session = get_session()
    job = Job(
        id=job_id,
        source_type="link",
        source_url=url,
        media_type="video",  # refined after download/extraction if needed
        target_language=target_language,
        voice=voice,
        status="pending",
        current_step_label="Queued",
    )
    session.add(job)
    session.commit()
    session.close()

    worker.submit_job(job_id)
    return {"job_id": job_id}


@app.get("/jobs")
def list_jobs():
    session = get_session()
    jobs = session.query(Job).order_by(Job.created_at.desc()).all()
    result = [_job_summary(j) for j in jobs]
    session.close()
    return result


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    session = get_session()
    job = session.query(Job).filter(Job.id == job_id).first()
    if not job:
        session.close()
        raise HTTPException(status_code=404, detail="Job not found")

    chunks = session.query(Chunk).filter(Chunk.job_id == job_id).order_by(Chunk.index).all()
    data = _job_summary(job)
    data["chunks"] = [
        {"index": c.index, "status": c.status, "start": c.start_time, "end": c.end_time}
        for c in chunks
    ]
    session.close()
    return data


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    session = get_session()
    job = session.query(Job).filter(Job.id == job_id).first()
    if not job:
        session.close()
        raise HTTPException(status_code=404, detail="Job not found")
    session.delete(job)
    session.commit()
    session.close()

    wd = os.path.join(STORAGE_DIR, job_id)
    if os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    return {"deleted": True}


@app.get("/jobs/{job_id}/download")
def download_result(job_id: str):
    session = get_session()
    job = session.query(Job).filter(Job.id == job_id).first()
    session.close()
    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="Result not ready")

    path = job.output_video_path or job.output_audio_path
    filename = os.path.basename(path)
    return FileResponse(path, filename=filename)


@app.get("/jobs/{job_id}/srt")
def download_srt(job_id: str):
    session = get_session()
    job = session.query(Job).filter(Job.id == job_id).first()
    session.close()
    if not job or not job.srt_path or not os.path.exists(job.srt_path):
        raise HTTPException(status_code=404, detail="Subtitles not ready")
    return FileResponse(job.srt_path, filename="subtitles.srt")


def _job_summary(job: Job) -> dict:
    return {
        "id": job.id,
        "source_type": job.source_type,
        "original_filename": job.original_filename,
        "source_url": job.source_url,
        "media_type": job.media_type,
        "source_language": job.source_language,
        "target_language": job.target_language,
        "voice": job.voice,
        "duration_seconds": job.duration_seconds,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "current_step_label": job.current_step_label,
        "error_message": job.error_message,
        "has_result": job.status == "done",
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }
