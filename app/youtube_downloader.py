"""
Downloads audio from a YouTube URL using yt-dlp and returns the local file path.
"""

import re
import yt_dlp
from pathlib import Path
from app.config import DOWNLOADS_DIR


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})',
        r'(?:watch\?.*v=)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_video_metadata(url: str) -> dict:
    """Fetch video title, channel, duration, and thumbnail without downloading."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Unknown"),
            "channel": info.get("channel", info.get("uploader", "Unknown")),
            "duration": info.get("duration", 0),
            "thumbnail": info.get("thumbnail", ""),
            "video_id": info.get("id", ""),
        }


def download_audio(url: str) -> Path:
    """
    Download the audio track from a YouTube video and convert to WAV.
    Returns the path to the downloaded WAV file.
    """
    video_id = extract_video_id(url)
    output_path = DOWNLOADS_DIR / f"{video_id}"

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "ffmpeg_location": r"C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Links",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    wav_path = output_path.with_suffix(".wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"Expected WAV file not found: {wav_path}")

    return wav_path
