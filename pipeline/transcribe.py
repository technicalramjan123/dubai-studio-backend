"""
Speech-to-text using faster-whisper (CTranslate2-based reimplementation
of OpenAI's open-source Whisper models). Chosen over the original
openai-whisper package because it ships prebuilt wheels (nothing to
compile) and doesn't require torch/triton -- much more reliable to
build on minimal/free-tier hosting like Railway, and faster on CPU.
"""
import os
from faster_whisper import WhisperModel

_model = None
_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
_compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # int8 = lowest RAM usage


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(_model_size, device="cpu", compute_type=_compute_type)
    return _model


def transcribe_chunk(audio_path: str, source_language_hint: str = None):
    """
    Returns: {"text": str, "language": str}
    source_language_hint: ISO code like "en" to skip auto-detection on
    subsequent chunks once the language is known from chunk 1.
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        language=source_language_hint,  # None = auto-detect
        vad_filter=True,  # skip silence, slightly faster and cleaner text
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    detected_language = info.language if info and info.language else (source_language_hint or "unknown")

    return {"text": text, "language": detected_language}


def transcribe_chunk_segments(audio_path: str, source_language_hint: str = None):
    """
    Sentence-level transcription: returns each spoken segment with its own
    start/end timestamps (relative to the start of this audio chunk), so
    each sentence's dub can later be placed at the correct moment instead
    of stretching a whole multi-minute chunk uniformly.

    Returns: {"segments": [{"start": float, "end": float, "text": str}, ...], "language": str}
    """
    model = get_model()
    segments_iter, info = model.transcribe(
        audio_path,
        language=source_language_hint,
        vad_filter=True,
    )
    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})

    detected_language = info.language if info and info.language else (source_language_hint or "unknown")
    return {"segments": segments, "language": detected_language}
