"""
Wrapper around yt-dlp for pulling video/audio from a pasted link
(YouTube or a direct .mp4/.mp3 URL).
"""
import subprocess
import glob
import os


class DownloadError(Exception):
    pass


def download_from_url(url: str, out_dir: str) -> str:
    """Downloads the best available video+audio (or audio-only) stream.
    Returns the path to the downloaded file."""
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "source.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--no-playlist",
        "-o", out_template,
        url,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise DownloadError(f"yt-dlp failed: {result.stderr[-2000:]}")

    matches = glob.glob(os.path.join(out_dir, "source.*"))
    if not matches:
        raise DownloadError("yt-dlp reported success but no output file was found.")
    return matches[0]
