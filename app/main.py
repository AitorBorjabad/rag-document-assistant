import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from groq import Groq
from app.rag_engine import process_and_index_pdf, search_relevant_chunks
from app.schemas import AnswerResponse, QuestionRequest, SourceChunk, UploadResponse

load_dotenv()

app = FastAPI(
    title="RAG Document Assistant API",
    description="API de consulta documental con búsqueda vectorial y citas verificables.",
    version="1.0.0",
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Solo se admiten archivos PDF."
        )

    content = await file.read()
    chunks_count = process_and_index_pdf(content, file.filename)

    return UploadResponse(
        filename=file.filename,
        chunks_indexed=chunks_count,
        message="Documento procesado e indexado correctamente.",
    )


@app.post("/chat/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    relevant_chunks = search_relevant_chunks(request.question, top_k=3)

    if not relevant_chunks:
        raise HTTPException(
            status_code=404, detail="No hay documentos indexados aún."
        )

    context_text = "\n\n".join(
        [f"[Página {c['page']}]: {c['text']}" for c in relevant_chunks]
    )

    system_prompt = (
        "Eres un asistente técnico preciso. Responde a la pregunta del usuario "
        "basándote ÚNICAMENTE en el contexto proporcionado. Si la respuesta no está "
        "en el contexto, di 'No encuentro esa información en el documento'."
    )

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Contexto:\n{context_text}\n\nPregunta: {request.question}",
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    answer = chat_completion.choices[0].message.content
    sources = [
        SourceChunk(page=c["page"], text_snippet=c["text"][:150] + "...")
        for c in relevant_chunks
    ]

    return AnswerResponse(answer=answer, sources=sources)
