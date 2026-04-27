"""
LangChain-powered Q&A chain that retrieves relevant chunks from Pinecone
and generates answers using Google Gemini LLM.
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from pinecone import Pinecone

from app.config import (
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

# ── Prompt Template ─────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent assistant that answers questions about a YouTube video
based on its transcript. Use the provided transcript excerpts to answer accurately.

If the answer is not found in the transcript, say so clearly — do not make things up.
When possible, reference the relevant part of the video.

Transcript excerpts:
{context}

Question: {question}

Answer:""",
)


def build_qa_chain(video_id: str):
    """
    Build a conversational Q&A chain for a specific video namespace.

    Returns:
        (chain, memory) tuple so callers can manage chat history.
    """
    # Pinecone retriever
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=video_id,
        text_key="text",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # LLM — Google Gemini
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    # Conversation memory (keep last 5 exchanges)
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )

    return chain, memory


def ask_question(chain, question: str) -> dict:
    """
    Ask a question using the conversational chain.

    Returns:
        {
            "answer": "...",
            "sources": ["chunk text 1", "chunk text 2", ...]
        }
    """
    result = chain.invoke({"question": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append(doc.page_content)

    return {
        "answer": result["answer"],
        "sources": sources,
    }
