"""
Speech-to-text using OpenAI's open-source Whisper model, run locally
(no paid API). Uses the "base" model by default — a good balance of
accuracy and memory/CPU usage for free-tier hosting. Set the
WHISPER_MODEL_SIZE env var to "tiny" for even lighter resource usage,
or "small"/"medium" if the host has more RAM available.
"""
import os
import whisper

_model = None
_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")


def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(_model_size)
    return _model


def transcribe_chunk(audio_path: str, source_language_hint: str = None):
    """
    Returns: {"text": str, "language": str}
    source_language_hint: ISO code like "en" to skip auto-detection on
    subsequent chunks once the language is known from chunk 1.
    """
    model = get_model()
    kwargs = {}
    if source_language_hint:
        kwargs["language"] = source_language_hint

    result = model.transcribe(audio_path, fp16=False, **kwargs)
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", source_language_hint or "unknown"),
    }
