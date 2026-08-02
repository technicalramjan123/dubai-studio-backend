"""
Orchestrates a dubbing job end to end:
  download/locate source -> extract audio -> split into silence-aware
  chunks -> per chunk: sentence-level transcribe -> translate each
  sentence -> synthesize each sentence's voice -> place each dubbed
  sentence at its correct timestamp (silence-padded) -> stitch chunks
  together -> (if video) merge with original video.

Sentence-level placement (rather than treating a whole multi-minute
chunk as one blob) is what keeps the dubbed voice in sync with the
original video's pacing, instead of drifting or sounding artificially
slowed down.

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
    logger.info(f"[{job.id}] {status} - {label} ({job.progress_percent:.0f}%)")


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
    all_segments_global = []  # for accurate, sentence-level subtitles

    for i, chunk in enumerate(chunks):
        pct = 15 + (i / max(total_chunks, 1)) * 70
        set_status(session, job, "processing",
                   f"Processing chunk {i + 1} of {total_chunks} (sentence-by-sentence)...", pct)

        if chunk.status == "done" and chunk.audio_chunk_path and os.path.exists(chunk.audio_chunk_path):
            translated_audio_paths.append(chunk.audio_chunk_path)
            if chunk.segments_json:
                for seg in json.loads(chunk.segments_json):
                    all_segments_global.append({
                        "start": chunk.start_time + seg["start"],
                        "end": chunk.start_time + seg["end"],
                        "source_text": seg["source_text"],
                        "translated_text": seg["translated_text"],
                    })
            continue

        attempt = 0
        last_error = None
        while attempt <= MAX_CHUNK_RETRIES:
            try:
                chunk_wav = os.path.join(wd, f"chunk_{chunk.index}.wav")
                extract.split_audio_chunk(audio_path, chunk.start_time, chunk.end_time, chunk_wav)

                result = transcribe.transcribe_chunk_segments(chunk_wav, source_language_hint=detected_source_lang)
                segments = result["segments"]
                if not detected_source_lang:
                    detected_source_lang = result["language"]
                    job.source_language = detected_source_lang
                    session.commit()

                chunk_duration = chunk.end_time - chunk.start_time
                parts = []
                enriched_segments = []
                prev_end = 0.0

                if not segments:
                    silence_path = os.path.join(wd, f"chunk_{chunk.index}_allsilence.mp3")
                    extract.make_silence(chunk_duration, silence_path)
                    chunk_audio_path = silence_path
                else:
                    total_segments = len(segments)
                    chunk_pct_span = 70 / max(total_chunks, 1)
                    audio_cursor = 0.0  # actual elapsed time in the dubbed track so far

                    for s_idx, seg in enumerate(segments):
                        set_status(
                            session, job, "processing",
                            f"Chunk {i + 1}/{total_chunks} — sentence {s_idx + 1}/{total_segments}...",
                            pct + (s_idx / max(total_segments, 1)) * chunk_pct_span
                        )

                        # Only insert silence if the dubbed track's actual
                        # position is still behind the original speaker's
                        # start time. If a previous sentence ran long, skip
                        # the gap instead of rewinding — we never cut audio,
                        # so occasionally running a little behind is the
                        # honest trade-off.
                        gap = seg["start"] - audio_cursor
                        if gap > 0.08:
                            gap_path = os.path.join(wd, f"chunk_{chunk.index}_gap_{s_idx}.mp3")
                            extract.make_silence(gap, gap_path)
                            parts.append(gap_path)
                            audio_cursor += gap

                        try:
                            translated_text = translate.translate_text(
                                seg["text"], detected_source_lang, job.target_language)
                        except Exception:
                            translated_text = seg["text"]  # fall back to untranslated rather than failing the chunk

                        raw_tts_path = os.path.join(wd, f"chunk_{chunk.index}_seg_{s_idx}_raw.mp3")
                        seg_duration_hint = max(0.2, seg["end"] - seg["start"])
                        try:
                            tts.synthesize(translated_text, job.target_language,
                                           job.voice or "male", raw_tts_path)
                        except Exception:
                            extract.make_silence(seg_duration_hint, raw_tts_path)  # skip this sentence's audio rather than failing the chunk

                        fitted_path = os.path.join(wd, f"chunk_{chunk.index}_seg_{s_idx}_fit.mp3")
                        extract.fit_audio_duration(raw_tts_path, seg_duration_hint, fitted_path)
                        parts.append(fitted_path)
                        audio_cursor += extract.get_duration_seconds(fitted_path)

                        enriched_segments.append({
                            "start": seg["start"], "end": seg["end"],
                            "source_text": seg["text"], "translated_text": translated_text,
                        })
                        prev_end = seg["end"]

                    trailing_gap = chunk_duration - audio_cursor
                    if trailing_gap > 0.08:
                        trail_path = os.path.join(wd, f"chunk_{chunk.index}_trail.mp3")
                        extract.make_silence(trailing_gap, trail_path)
                        parts.append(trail_path)
                    # If audio_cursor overran chunk_duration (translated
                    # speech needed more room than the original), we
                    # deliberately do NOT trim — the chunk's dubbed audio
                    # will simply be a little longer than the original
                    # chunk, which is far better than cutting words off.

                    chunk_audio_path = os.path.join(wd, f"chunk_{chunk.index}_final.mp3")
                    extract.concat_audio_chunks(parts, chunk_audio_path, wd)

                chunk.source_text = " ".join(s["text"] for s in segments)
                chunk.translated_text = " ".join(s["translated_text"] for s in enriched_segments)
                chunk.segments_json = json.dumps(enriched_segments, ensure_ascii=False)
                chunk.audio_chunk_path = chunk_audio_path
                chunk.status = "done"
                chunk.error_message = None
                session.commit()

                translated_audio_paths.append(chunk_audio_path)
                for seg in enriched_segments:
                    all_segments_global.append({
                        "start": chunk.start_time + seg["start"],
                        "end": chunk.start_time + seg["end"],
                        "source_text": seg["source_text"],
                        "translated_text": seg["translated_text"],
                    })
                break  # chunk succeeded

            except Exception as e:
                attempt += 1
                last_error = str(e)
                chunk.retry_count = attempt
                chunk.error_message = last_error
                session.commit()
                logger.warning(f"[{job.id}] chunk {chunk.index} attempt {attempt} failed: {e}")

        else:
            chunk.status = "failed"
            session.commit()
            raise RuntimeError(f"Chunk {chunk.index} failed after {MAX_CHUNK_RETRIES} retries: {last_error}")

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

    # --- Step 7: save transcript / subtitles (sentence-level, accurate timing) ---
    all_segments_global.sort(key=lambda s: s["start"])
    job.transcript_json = json.dumps(all_segments_global, ensure_ascii=False)
    srt_path = os.path.join(wd, "subtitles.srt")
    _write_srt(all_segments_global, srt_path)
    job.srt_path = srt_path

    set_status(session, job, "done", "Completed", 100)


def _srt_timestamp(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['translated_text'] or ''}\n\n")
