"""
Translation using free services only:
  1. LibreTranslate (self-hosted URL via LIBRETRANSLATE_URL env var,
     or a public instance) as the primary engine.
  2. Google Translate's free unofficial endpoint as a fallback if
     LibreTranslate fails or is unavailable — keeps the job moving
     instead of failing the whole chunk.
"""
import os
import time
import requests

LIBRETRANSLATE_URL = os.environ.get(
    "LIBRETRANSLATE_URL", "https://libretranslate.com/translate"
)
LIBRETRANSLATE_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY", "")


class TranslationError(Exception):
    pass


def _translate_libretranslate(text: str, source: str, target: str) -> str:
    payload = {
        "q": text,
        "source": source if source else "auto",
        "target": target,
        "format": "text",
    }
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY

    resp = requests.post(LIBRETRANSLATE_URL, data=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "translatedText" not in data:
        raise TranslationError(f"Unexpected LibreTranslate response: {data}")
    return data["translatedText"]


def _translate_google_free(text: str, source: str, target: str) -> str:
    """Uses Google's free/unofficial translate_a/single endpoint (no key required)."""
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": source if source else "auto",
        "tl": target,
        "dt": "t",
        "q": text,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    translated = "".join(segment[0] for segment in data[0] if segment[0])
    return translated


def translate_text(text: str, source_lang: str, target_lang: str,
                    retries: int = 2) -> str:
    if not text.strip():
        return ""

    last_error = None
    for attempt in range(retries + 1):
        try:
            return _translate_libretranslate(text, source_lang, target_lang)
        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))

    # Fallback engine
    try:
        return _translate_google_free(text, source_lang, target_lang)
    except Exception as e:
        raise TranslationError(
            f"Both translation engines failed. LibreTranslate error: {last_error}; "
            f"Google fallback error: {e}"
        )
