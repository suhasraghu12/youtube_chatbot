"""
FastAPI application — YouTube Video Q&A API.

Endpoints:
    POST /api/process    — Accept a YouTube URL, download, transcribe, embed
    POST /api/ask        — Ask a question about a processed video
    GET  /api/status     — Check processing status
    GET  /               — Serve the frontend
"""

import asyncio
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from app.youtube_downloader import download_audio, get_video_metadata, extract_video_id
from app.transcriber import transcribe_audio, segments_to_timestamped_text
from app.embedder import embed_and_store, delete_video_vectors
from app.qa_chain import build_qa_chain, ask_question

# ── In-memory state ────────────────────────────────────────────────────────
# Tracks processing jobs and active Q&A chains
processing_jobs: dict = {}   # job_id -> {status, progress, video_id, metadata, ...}
active_chains: dict = {}     # video_id -> (chain, memory)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup / shutdown hooks."""
    print("[*] YouTube Q&A API starting up...")
    yield
    # Cleanup downloaded files on shutdown
    downloads = Path(__file__).resolve().parent / "downloads"
    if downloads.exists():
        for f in downloads.iterdir():
            f.unlink(missing_ok=True)
    print("[*] Shutting down -- cleaned up downloads.")


app = FastAPI(
    title="YouTube Video Q&A",
    description="Transcribe YouTube videos with Whisper & ask questions via LangChain + Pinecone",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Pydantic Models ────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    url: str

class AskRequest(BaseModel):
    video_id: str
    question: str

class ProcessResponse(BaseModel):
    job_id: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    video_id: str | None = None
    metadata: dict | None = None
    transcript_preview: str | None = None
    chunks_count: int | None = None
    error: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Background Processing ──────────────────────────────────────────────────

async def _process_video(job_id: str, url: str):
    """Run the full pipeline: download → transcribe → embed."""
    job = processing_jobs[job_id]

    try:
        # Step 1: Fetch metadata
        job["status"] = "fetching_metadata"
        job["progress"] = 10
        meta = await asyncio.to_thread(get_video_metadata, url)
        job["metadata"] = meta
        job["video_id"] = meta["video_id"]

        # Step 2: Download audio
        job["status"] = "downloading"
        job["progress"] = 25
        audio_path = await asyncio.to_thread(download_audio, url)

        # Step 3: Transcribe
        job["status"] = "transcribing"
        job["progress"] = 50
        result = await asyncio.to_thread(transcribe_audio, audio_path)
        job["transcript_preview"] = result["text"][:500] + ("..." if len(result["text"]) > 500 else "")
        job["full_transcript"] = result["text"]
        job["segments"] = result["segments"]

        # Step 4: Chunk & Embed
        job["status"] = "embedding"
        job["progress"] = 75
        timestamped = segments_to_timestamped_text(result["segments"])
        n_chunks = await asyncio.to_thread(
            embed_and_store,
            meta["video_id"],
            timestamped,
            {"title": meta["title"], "channel": meta["channel"]},
        )
        job["chunks_count"] = n_chunks

        # Step 5: Build Q&A chain
        job["status"] = "ready"
        job["progress"] = 100
        chain, memory = build_qa_chain(meta["video_id"])
        active_chains[meta["video_id"]] = (chain, memory)

        # Cleanup audio file
        audio_path.unlink(missing_ok=True)

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[Pipeline] Error processing {url}: {e}")


# ── API Routes ──────────────────────────────────────────────────────────────

@app.post("/api/process", response_model=ProcessResponse)
async def process_video(req: ProcessRequest):
    """Start processing a YouTube video."""
    # Validate URL
    url = req.url.strip()
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(400, "Please provide a valid YouTube URL.")

    try:
        video_id = extract_video_id(url)
    except ValueError:
        raise HTTPException(400, "Could not extract video ID from the URL.")

    # Check if already processed
    if video_id in active_chains:
        return ProcessResponse(
            job_id=video_id,
            message="Video already processed! You can start asking questions.",
        )

    # Create job and start background task
    job_id = str(uuid.uuid4())[:8]
    processing_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "video_id": None,
        "metadata": None,
        "transcript_preview": None,
        "chunks_count": None,
        "error": None,
    }
    asyncio.create_task(_process_video(job_id, url))

    return ProcessResponse(job_id=job_id, message="Processing started!")


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Check the status of a processing job."""
    # Check if job_id is a video_id that was already processed
    if job_id in active_chains:
        return StatusResponse(
            job_id=job_id,
            status="ready",
            progress=100,
            video_id=job_id,
        )

    job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    return StatusResponse(job_id=job_id, **job)


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Ask a question about a processed video."""
    if req.video_id not in active_chains:
        raise HTTPException(400, "Video not processed yet. Please process it first.")

    chain, _ = active_chains[req.video_id]
    result = await asyncio.to_thread(ask_question, chain, req.question)

    return AskResponse(**result)


@app.get("/api/transcript/{video_id}")
async def get_transcript(video_id: str):
    """Return the full transcript for a processed video."""
    for job in processing_jobs.values():
        if job.get("video_id") == video_id and job.get("full_transcript"):
            return {
                "transcript": job["full_transcript"],
                "segments": job.get("segments", []),
            }
    raise HTTPException(404, "Transcript not found for this video.")


@app.post("/api/reset/{video_id}")
async def reset_video(video_id: str):
    """Delete embeddings and reset the chain for a video."""
    if video_id in active_chains:
        del active_chains[video_id]

    try:
        await asyncio.to_thread(delete_video_vectors, video_id)
    except Exception:
        pass

    return {"message": f"Video {video_id} has been reset."}


# ── Static Files & Frontend ────────────────────────────────────────────────

frontend_dir = Path(__file__).resolve().parent / "static"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "YouTube Q&A API is running. Frontend not found at /static/index.html"}
