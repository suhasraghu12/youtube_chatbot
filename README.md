# YouTube Video Q&A 🎬🤖

AI-powered YouTube video Q&A chatbot. Paste any YouTube link — the app transcribes it with **Whisper**, embeds the transcript into **Pinecone**, and lets you ask questions about the video using **LangChain + GPT**.

![Stack](https://img.shields.io/badge/Whisper-Transcription-green?style=flat-square)
![Stack](https://img.shields.io/badge/LangChain-RAG-blue?style=flat-square)
![Stack](https://img.shields.io/badge/Pinecone-VectorDB-purple?style=flat-square)
![Stack](https://img.shields.io/badge/FastAPI-Backend-red?style=flat-square)

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
┌───────────────┐
│  yt-dlp       │  Download audio track
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Whisper       │  Transcribe audio → timestamped text
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  LangChain     │  Chunk text (1000 chars, 200 overlap)
│  + OpenAI      │  Generate embeddings (text-embedding-3-small)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Pinecone      │  Store vectors in serverless index
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Q&A Chain     │  Retrieve relevant chunks → GPT-4o-mini → Answer
│  (LangChain)   │  With conversational memory (5-turn window)
└───────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and in your PATH ([download](https://ffmpeg.org/download.html))
- **OpenAI API key** ([get one](https://platform.openai.com/api-keys))
- **Pinecone API key** ([sign up free](https://www.pinecone.io/))

### 2. Install Dependencies

```bash
cd "YouTube Chatbot"
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example and fill in your keys
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-actual-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=youtube-qa
WHISPER_MODEL_SIZE=base
```

### 4. Run the Application

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

---

## 📁 Project Structure

```
YouTube Chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py                # Environment & settings
│   ├── main.py                  # FastAPI application & routes
│   ├── youtube_downloader.py    # yt-dlp audio download
│   ├── transcriber.py           # Whisper transcription
│   ├── embedder.py              # Text chunking + Pinecone storage
│   ├── qa_chain.py              # LangChain Q&A chain
│   └── static/
│       ├── index.html           # Frontend UI
│       ├── styles.css           # Premium dark theme
│       └── app.js               # Frontend logic
├── downloads/                   # Temp audio files (auto-cleaned)
├── .env                         # API keys (not committed)
├── .env.example                 # Template for .env
├── requirements.txt             # Python dependencies
├── run.py                       # Entry point
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process` | Start processing a YouTube URL |
| `GET` | `/api/status/{job_id}` | Check processing progress |
| `POST` | `/api/ask` | Ask a question about a processed video |
| `GET` | `/api/transcript/{video_id}` | Get full transcript |
| `POST` | `/api/reset/{video_id}` | Clear embeddings & reset |
| `GET` | `/` | Serve the frontend |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for answers |
| `LLM_TEMPERATURE` | `0.2` | Response creativity (0=deterministic) |

---

## 💡 How It Works

1. **Download**: `yt-dlp` fetches the audio track from YouTube and converts to WAV
2. **Transcribe**: OpenAI Whisper processes the audio into timestamped text segments
3. **Chunk & Embed**: LangChain splits the transcript into overlapping chunks, OpenAI generates embeddings
4. **Store**: Vectors are upserted into Pinecone under a namespace = video ID
5. **Q&A**: When you ask a question, LangChain retrieves the 5 most relevant chunks and feeds them to GPT-4o-mini for a grounded answer
6. **Memory**: The conversation maintains a 5-turn sliding window for follow-up questions

---

## 📝 License

MIT
