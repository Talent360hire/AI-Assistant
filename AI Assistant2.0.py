# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import traceback

# PDF/text processing
# Try to import pdfplumber; if not available, fall back to None and use PyPDF2 at runtime.
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# embeddings & vector DB
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM (local via ollama)
import ollama

# simple text splitter
import math

# Initialize FastAPI
app = FastAPI(title="OpenAIHub - RAG Assistant", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model (sentence-transformers)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good for retrieval
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Initialize Chroma DB (local persistence folder ./chroma_db)
CHROMA_DIR = "./chroma_db"
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))

# Use or create a collection for documents
COLLECTION_NAME = "openaihub_docs"
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=COLLECTION_NAME)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_parts = []

    # write bytes to a temporary file first (both pdfplumber and PyPDF2 work with filenames)
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmpf.write(pdf_bytes)
        tmpf.flush()
        tmpf.close()

        # Primary: use pdfplumber if available
        if pdfplumber is not None:
            try:
                with pdfplumber.open(tmpf.name) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
            except Exception:
                # if pdfplumber fails, we'll try PyPDF2 below
                pass

        # Fallback: try PyPDF2 if no text was extracted yet
        if not text_parts:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(tmpf.name)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                    except Exception:
                        page_text = None
                    if page_text:
                        text_parts.append(page_text)
            except Exception:
                # final fallback: return empty string
                pass
    finally:
        try:
            os.unlink(tmpf.name)
        except Exception:
            pass

    return "\n\n".join(text_parts)
        except Exception:
            pass

    return "\n\n".join(text_parts)

def chunk_text(text: str, max_tokens=500, overlap=50):
    """
    Simple chunker by characters. max_tokens approximated by characters.
    Returns list of text chunks.
    """
    if not text:
        return []
    # approximate token by characters (conservative)
    max_chars = max_tokens * 4  # rough approximation (1 token ≈ 4 chars)
    overlap_chars = overlap * 4
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        if end >= L:
            chunks.append(text[start:L].strip())
            break
        # try to break at newline or sentence end
        slice_text = text[start:end]
        # prefer last newline
        nl = slice_text.rfind("\n")
        if nl > int(max_chars * 0.4):
            split_at = start + nl
        else:
            # fallback to last space
            sp = slice_text.rfind(" ")
            split_at = start + sp if sp > 0 else end
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = split_at - overlap_chars
        if start < 0:
            start = 0
        # avoid infinite loops
        if len(chunks) > 1000:
            break
    return chunks

# API models
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "OpenAIHub RAG Assistant — backend up"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file. The endpoint extracts text,
    splits into chunks, computes embeddings, and upserts into Chroma.
    Returns a summary of inserted chunks.
    """
    text = ""
    # read file contents and extract text:
    contents = await file.read()
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf_bytes(contents)
    else:
        # treat as text
        try:
            text = contents.decode("utf-8")
        except Exception:
            text = contents.decode("latin-1", errors="ignore")

    if not text or len(text.strip()) == 0:
        return {"error": "No text extracted from uploaded file."}

    # create chunks
    chunks = chunk_text(text, max_tokens=500, overlap=50)
    if len(chunks) == 0:
        return {"error": "Failed to split document into chunks."}

    # compute embeddings in batch
    embeddings = embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True).tolist()

    # generate ids and metadata
    ids = [f"{file.filename}__{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": file.filename,
            "chunk_index": i,
            "text_preview": (chunks[i][:200] + "...") if len(chunks[i]) > 200 else chunks[i]
        }
        for i in range(len(chunks))
    ]

    # upsert into chroma
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

    # persist DB
    try:
        client.persist()
    except Exception:
        # not fatal
        pass

    return {"filename": file.filename, "chunks_added": len(chunks)}
@app.post("/chat")
def chat(request: ChatRequest): 
    """
    Chat endpoint. Given a user message, retrieves relevant document chunks
    from Chroma, constructs a prompt, and queries the local LLM via Ollama.
    Returns the LLM response.
    """
    user_message = request.message

    # embed user message
    user_embedding = embed_model.encode([user_message], convert_to_numpy=True).tolist()[0]

    # retrieve top-k relevant chunks
    results = collection.query(
        query_embeddings=[user_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []

    # construct context
    context_texts = []
    for i, chunk in enumerate(retrieved_chunks):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")
        context_texts.append(f"[Source: {source}]\n{chunk}")

    context = "\n\n".join(context_texts)

    # construct prompt
    prompt = f"You are an AI assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_message}\n\nAnswer:"

    # query local LLM via Ollama
    try:
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        answer = f"Error querying LLM: {str(e)}"

    return {"answer": answer}