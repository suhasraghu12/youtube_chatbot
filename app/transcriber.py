"""
Transcribes audio files using OpenAI Whisper (local model).
Returns timestamped segments for richer context.
"""

import whisper
from pathlib import Path
from app.config import WHISPER_MODEL_SIZE

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Load the Whisper model (cached after first call)."""
    global _model
    if _model is None:
        print(f"[Whisper] Loading '{WHISPER_MODEL_SIZE}' model...")
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
        print("[Whisper] Model loaded successfully.")
    return _model


def transcribe_audio(audio_path: Path) -> dict:
    """
    Transcribe an audio file and return the full result including segments.

    Returns:
        {
            "text": "full transcript...",
            "segments": [
                {"start": 0.0, "end": 5.2, "text": "..."},
                ...
            ],
            "language": "en"
        }
    """
    model = _get_model()
    print(f"[Whisper] Transcribing: {audio_path.name}")

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        fp16=False,  # Use FP32 for CPU compatibility
    )

    print(f"[Whisper] Transcription complete — {len(result['segments'])} segments")

    return {
        "text": result["text"].strip(),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
        "language": result.get("language", "en"),
    }


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def segments_to_timestamped_text(segments: list) -> str:
    """Convert segments to a readable timestamped transcript."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)
