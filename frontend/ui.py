import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Asistente Documental RAG", page_icon="📚", layout="centered")
st.title("📚 Asistente RAG de Documentación")
st.caption("Sube un PDF y haz preguntas con respuestas verificables basadas en el documento.")

with st.sidebar:
    st.header("1. Subir Documento")
    uploaded_file = st.file_uploader("Selecciona un archivo PDF", type=["pdf"])
    
    if uploaded_file is not None and st.button("Procesar e Indexar PDF"):
        with st.spinner("Extrayendo texto y generando embeddings..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{API_URL}/documents/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"✅ Indexados {data['chunks_indexed']} fragmentos de '{data['filename']}'.")
                else:
                    st.error(f"Error: {res.json().get('detail', 'Error desconocido')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ No se puede conectar con el backend. Asegúrate de que FastAPI esté corriendo.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Ver fuentes citadas"):
                for src in msg["sources"]:
                    st.markdown(f"**Página {src['page']}:** _{src['text_snippet']}_")

if prompt := st.chat_input("Escribe tu pregunta sobre el documento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando contexto y generando respuesta..."):
            try:
                res = requests.post(f"{API_URL}/chat/ask", json={"question": prompt})
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    st.markdown(answer)
                    with st.expander("Ver fuentes citadas"):
                        for src in sources:
                            st.markdown(f"**Página {src['page']}:** _{src['text_snippet']}_")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                else:
                    st.warning(res.json().get("detail", "No se pudo obtener respuesta."))
            except requests.exceptions.ConnectionError:
                st.error("❌ Error de conexión con la API.")
