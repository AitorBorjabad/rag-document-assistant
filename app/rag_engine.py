import io
import json
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

INDEX_PATH = "./data/faiss_index.bin"
METADATA_PATH = "./data/metadata.json"

if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        documents_store = json.load(f)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    documents_store = []


def _chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_and_index_pdf(file_bytes: bytes, filename: str) -> int:
    reader = PdfReader(io.BytesIO(file_bytes))
    total_chunks = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        chunks = _chunk_text(text)
        embeddings = embedder.encode(chunks)
        embeddings_np = np.array(embeddings, dtype="float32")

        index.add(embeddings_np)

        for chunk in chunks:
            documents_store.append(
                {"text": chunk, "source": filename, "page": page_num}
            )

        total_chunks += len(chunks)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(documents_store, f, ensure_ascii=False, indent=2)

    return total_chunks


def search_relevant_chunks(question: str, top_k: int = 3) -> list[dict]:
    if index.ntotal == 0:
        return []

    question_embedding = np.array(
        embedder.encode([question]), dtype="float32"
    )
    k = min(top_k, index.ntotal)
    _, indices = index.search(question_embedding, k)

    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(documents_store):
            retrieved.append(documents_store[idx])

    return retrieved
