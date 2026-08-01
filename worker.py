"""
Orchestrates a dubbing job end to end:
  download/locate source -> extract audio -> split into silence-aware
  chunks -> per chunk: transcribe -> translate -> synthesize ->
  stitch chunks back together -> (if video) merge with original video.

Runs in a background thread per job via a simple in-process queue
(ThreadPoolExecutor) — no external queue service needed, which keeps
this deployable on Railway's free tier without extra infrastructure.

Designed to be resumable: each chunk's status is persisted, so if the
process restarts mid-job, only unfinished chunks are reprocessed.
"""
import os
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from models import get_session, Job, Chunk
from pipeline import extract, download, transcribe, translate, tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")
MAX_WORKERS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
MAX_CHUNK_RETRIES = 2

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def job_dir(job_id: str) -> str:
    d = os.path.join(STORAGE_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def set_status(session, job: Job, status: str, label: str, percent: float = None):
    job.status = status
    job.current_step_label = label
    if percent is not None:
        job.progress_percent = percent
    session.commit()
    logger.info(f"[{job.id}] {status} — {label} ({job.progress_percent:.0f}%)")


def submit_job(job_id: str):
    executor.submit(_process_job_safe, job_id)


def _process_job_safe(job_id: str):
    session = get_session()
    try:
        job = session.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        _process_job(session, job)
    except Exception as e:
        logger.error(f"[{job_id}] FAILED: {e}\n{traceback.format_exc()}")
        job = session.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            session.commit()
    finally:
        session.close()


def _process_job(session, job: Job):
    wd = job_dir(job.id)

    # --- Step 1: obtain source file ---
    if job.status in ("pending",):
        if job.source_type == "link":
            set_status(session, job, "downloading", "Downloading from link...", 2)
            try:
                job.input_path = download.download_from_url(job.source_url, wd)
            except Exception as e:
                raise RuntimeError(f"Download failed: {e}")
        set_status(session, job, "extracting", "Extracting audio...", 8)

    # --- Step 2: extract audio + measure duration ---
    audio_path = os.path.join(wd, "full_audio.wav")
    if not os.path.exists(audio_path):
        extract.extract_audio(job.input_path, audio_path)
    duration = extract.get_duration_seconds(audio_path)
    job.duration_seconds = duration
    session.commit()

    # --- Step 3: build chunk plan (only once) ---
    existing_chunks = session.query(Chunk).filter(Chunk.job_id == job.id).count()
    if existing_chunks == 0:
        set_status(session, job, "extracting", "Planning chunks (detecting pauses)...", 12)
        silences = extract.detect_silences(audio_path)
        boundaries = extract.compute_chunk_boundaries(duration, silences)
        for idx, (start, end) in enumerate(boundaries):
            session.add(Chunk(job_id=job.id, index=idx, start_time=start,
                               end_time=end, status="pending"))
        session.commit()

    chunks = (session.query(Chunk).filter(Chunk.job_id == job.id)
              .order_by(Chunk.index).all())
    total_chunks = len(chunks)

    # --- Step 4: process each chunk (skip ones already done -> resumable) ---
    detected_source_lang = job.source_language
    translated_audio_paths = []

    for i, chunk in enumerate(chunks):
        pct = 15 + (i / max(total_chunks, 1)) * 70
        set_status(session, job, "processing",
                   f"Processing chunk {i + 1} of {total_chunks}...", pct)

        if chunk.status == "done" and chunk.audio_chunk_path and os.path.exists(chunk.audio_chunk_path):
            translated_audio_paths.append(chunk.audio_chunk_path)
            continue

        try:
            chunk_wav = os.path.join(wd, f"chunk_{chunk.index}.wav")
            extract.split_audio_chunk(audio_path, chunk.start_time, chunk.end_time, chunk_wav)

            # Transcribe
            t = transcribe.transcribe_chunk(chunk_wav, source_language_hint=detected_source_lang)
            chunk.source_text = t["text"]
            if not detected_source_lang:
                detected_source_lang = t["language"]
                job.source_language = detected_source_lang
            chunk.status = "transcribed"
            session.commit()

            # Translate
            translated = translate.translate_text(
                t["text"], detected_source_lang, job.target_language
            )
            chunk.translated_text = translated
            chunk.status = "translated"
            session.commit()

            # Synthesize
            chunk_tts_path = os.path.join(wd, f"chunk_{chunk.index}_dub.mp3")
            tts.synthesize(translated, job.target_language, job.voice or "male", chunk_tts_path)

            # Stretch/compress the generated speech to match the original
            # chunk's exact duration so the dub doesn't drift out of sync
            # with the video as chunks accumulate.
            target_duration = chunk.end_time - chunk.start_time
            fitted_path = os.path.join(wd, f"chunk_{chunk.index}_fitted.mp3")
            extract.fit_audio_duration(chunk_tts_path, target_duration, fitted_path)

            chunk.audio_chunk_path = fitted_path
            chunk.status = "done"
            chunk.error_message = None
            session.commit()

            translated_audio_paths.append(fitted_path)

        except Exception as e:
            chunk.retry_count += 1
            chunk.error_message = str(e)
            session.commit()
            if chunk.retry_count <= MAX_CHUNK_RETRIES:
                logger.warning(f"[{job.id}] chunk {chunk.index} failed, will retry: {e}")
                try:
                    # one retry inline
                    chunk_tts_path = os.path.join(wd, f"chunk_{chunk.index}_dub.mp3")
                    if not chunk.translated_text:
                        t = transcribe.transcribe_chunk(chunk_wav, source_language_hint=detected_source_lang)
                        chunk.source_text = t["text"]
                        chunk.translated_text = translate.translate_text(
                            t["text"], detected_source_lang, job.target_language)
                    tts.synthesize(chunk.translated_text, job.target_language,
                                    job.voice or "male", chunk_tts_path)

                    target_duration = chunk.end_time - chunk.start_time
                    fitted_path = os.path.join(wd, f"chunk_{chunk.index}_fitted.mp3")
                    extract.fit_audio_duration(chunk_tts_path, target_duration, fitted_path)

                    chunk.audio_chunk_path = fitted_path
                    chunk.status = "done"
                    chunk.error_message = None
                    session.commit()
                    translated_audio_paths.append(fitted_path)
                    continue
                except Exception as e2:
                    chunk.error_message = str(e2)
                    session.commit()
            chunk.status = "failed"
            session.commit()
            raise RuntimeError(f"Chunk {chunk.index} failed after retries: {chunk.error_message}")

    # --- Step 5: stitch chunks back together ---
    set_status(session, job, "merging", "Combining dubbed audio...", 88)
    final_audio = os.path.join(wd, "final_dubbed_audio.mp3")
    extract.concat_audio_chunks(translated_audio_paths, final_audio, wd)
    job.output_audio_path = final_audio

    # --- Step 6: merge with video if needed ---
    if job.media_type == "video":
        set_status(session, job, "merging", "Merging audio with video...", 94)
        final_video = os.path.join(wd, "final_dubbed_video.mp4")
        extract.merge_audio_with_video(job.input_path, final_audio, final_video)
        job.output_video_path = final_video

    # --- Step 7: save transcript / subtitles ---
    transcript = [
        {
            "index": c.index, "start": c.start_time, "end": c.end_time,
            "source_text": c.source_text, "translated_text": c.translated_text,
        }
        for c in chunks
    ]
    job.transcript_json = json.dumps(transcript, ensure_ascii=False)
    srt_path = os.path.join(wd, "subtitles.srt")
    _write_srt(transcript, srt_path)
    job.srt_path = srt_path

    set_status(session, job, "done", "Completed", 100)


def _srt_timestamp(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(transcript, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(transcript, start=1):
            f.write(f"{i}\n")
            f.write(f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['translated_text'] or ''}\n\n")
