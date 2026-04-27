"""
Application configuration — loads environment variables and sets defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root, overriding any stale inherited variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ── Google Gemini ───────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Pinecone ────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "youtube-qa-gemini")

# ── Whisper ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Add WinGet Links to PATH for ffmpeg
import os
winget_links = r"C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Links"
if winget_links not in os.environ.get("PATH", ""):
    os.environ["PATH"] = winget_links + os.pathsep + os.environ.get("PATH", "")

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000        # characters per chunk
CHUNK_OVERLAP: int = 200      # overlap between chunks

# ── Embedding (Google) ──────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_DIMENSION: int = 3072   # Google gemini-embedding-001 outputs 3072 dims

# ── LLM (Gemini) ────────────────────────────────────────────────────────────
LLM_MODEL: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.2
