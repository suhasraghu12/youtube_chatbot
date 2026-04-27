"""
Chunks the transcript text and creates embeddings using Google Gemini,
then upserts them into a Pinecone vector index.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from app.config import (
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)

# ── Pinecone client ────────────────────────────────────────────────────────
_pc = None
_index = None


def _get_pinecone_index():
    """Initialise Pinecone and return the index, creating it if necessary."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create the index if it doesn't exist
        existing = [idx.name for idx in _pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            print(f"[Pinecone] Creating index '{PINECONE_INDEX_NAME}'...")
            _pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("[Pinecone] Index created.")

        _index = _pc.Index(PINECONE_INDEX_NAME)
        print(f"[Pinecone] Connected to index '{PINECONE_INDEX_NAME}'")
    return _index


def _get_embeddings():
    """Return the Google Generative AI embeddings instance."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def chunk_text(text: str) -> list[str]:
    """Split transcript text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed_and_store(
    video_id: str,
    text: str,
    metadata: dict | None = None,
) -> int:
    """
    Chunk the transcript, embed each chunk, and upsert into Pinecone.

    Args:
        video_id: YouTube video ID used as the namespace.
        text:     Full transcript text.
        metadata: Extra metadata to attach to every vector.

    Returns:
        Number of vectors upserted.
    """
    index = _get_pinecone_index()
    embeddings = _get_embeddings()

    chunks = chunk_text(text)
    if not chunks:
        return 0

    print(f"[Embedder] {len(chunks)} chunks created from transcript")

    # Generate embeddings in batch
    vectors_data = embeddings.embed_documents(chunks)

    # Build upsert payload
    vectors = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors_data)):
        vec_meta = {
            "text": chunk,
            "chunk_index": i,
            "video_id": video_id,
        }
        if metadata:
            vec_meta.update(metadata)

        vectors.append({
            "id": f"{video_id}_chunk_{i}",
            "values": vec,
            "metadata": vec_meta,
        })

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch, namespace=video_id)

    print(f"[Embedder] Upserted {len(vectors)} vectors into namespace '{video_id}'")
    return len(vectors)


def delete_video_vectors(video_id: str):
    """Remove all vectors for a given video from Pinecone."""
    index = _get_pinecone_index()
    index.delete(delete_all=True, namespace=video_id)
    print(f"[Embedder] Deleted all vectors in namespace '{video_id}'")
