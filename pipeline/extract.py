"""
Audio extraction and chunking utilities.
Uses ffmpeg for extraction/merging and ffmpeg's silencedetect filter
to find natural pause points for splitting long audio into chunks
(so we never cut mid-word, and can process long files piece by piece).
"""
import subprocess
import re
import os

DEFAULT_CHUNK_TARGET_SECONDS = 420  # aim for ~7 minute chunks


def run(cmd: list):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
    return result


def get_duration_seconds(path: str) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ])
    return float(result.stdout.strip())


def extract_audio(input_path: str, output_wav_path: str):
    """Extract mono 16kHz WAV audio from a video or audio file (Whisper's preferred format)."""
    run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_wav_path
    ])
    return output_wav_path


def detect_silences(audio_path: str, noise_db="-30dB", min_silence_dur=0.5):
    """Returns a list of (start, end) tuples of silent regions using ffmpeg silencedetect."""
    result = subprocess.run([
        "ffmpeg", "-i", audio_path, "-af",
        f"silencedetect=noise={noise_db}:d={min_silence_dur}",
        "-f", "null", "-"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    text = result.stderr
    starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", text)]
    silences = list(zip(starts, ends[:len(starts)]))
    return silences


def compute_chunk_boundaries(total_duration: float, silences: list,
                              target_len=DEFAULT_CHUNK_TARGET_SECONDS):
    """
    Walks through the timeline and picks cut points at the silence closest to
    each target_len interval, so chunks are ~target_len seconds but always
    cut during a natural pause rather than mid-word.
    """
    if total_duration <= target_len * 1.3:
        return [(0.0, total_duration)]

    boundaries = [0.0]
    next_target = target_len
    silence_midpoints = [(s + e) / 2 for s, e in silences]

    while next_target < total_duration:
        candidates = [t for t in silence_midpoints if boundaries[-1] + 60 < t < next_target + 90]
        if candidates:
            cut = min(candidates, key=lambda t: abs(t - next_target))
        else:
            cut = next_target  # no silence found nearby, hard cut
        if cut - boundaries[-1] > 30:  # avoid tiny slivers
            boundaries.append(cut)
        next_target = boundaries[-1] + target_len

    boundaries.append(total_duration)
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    return chunks


def split_audio_chunk(audio_path: str, start: float, end: float, out_path: str):
    duration = end - start
    run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start), "-t", str(duration),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_path
    ])
    return out_path


def concat_audio_chunks(chunk_paths: list, out_path: str, work_dir: str):
    list_file = os.path.join(work_dir, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in chunk_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
        "-acodec", "libmp3lame", out_path
    ])
    return out_path


def merge_audio_with_video(video_path: str, audio_path: str, out_path: str):
    """Replace the audio track of a video with the newly generated dubbed audio."""
    run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        out_path
    ])
    return out_path
