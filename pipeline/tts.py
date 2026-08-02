"""
Text-to-speech using Microsoft Edge's free neural voices via the
edge-tts library. IMPORTANT: these are generic, pre-built voices —
this module intentionally does NOT do any voice cloning or matching
to the original speaker. Voice cloning needs heavy GPU processing
and is out of scope for free-tier hosting.
"""
import asyncio
import edge_tts

# Generic male/female voice per supported target language.
# Extend this map to add more languages.
VOICE_MAP = {
    "bn": {"male": "bn-IN-BashkarNeural", "female": "bn-IN-TanishaaNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "en": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "ar": {"male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
    "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "ur": {"male": "ur-PK-AsadNeural", "female": "ur-PK-UzmaNeural"},
}


def resolve_voice(target_lang: str, voice_gender: str = "male") -> str:
    lang_voices = VOICE_MAP.get(target_lang, VOICE_MAP["en"])
    return lang_voices.get(voice_gender, lang_voices["male"])


async def _synthesize_async(text: str, voice: str, out_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str, target_lang: str, voice_gender: str, out_path: str):
    if not text.strip():
        # Write a tiny silent placeholder rather than crashing the job.
        with open(out_path, "wb") as f:
            f.write(b"")
        return out_path

    voice = resolve_voice(target_lang, voice_gender)
    try:
        asyncio.run(asyncio.wait_for(_synthesize_async(text, voice, out_path), timeout=20))
    except asyncio.TimeoutError:
        raise RuntimeError(f"Edge-TTS timed out generating speech for: {text[:50]!r}")
    return out_path
