# 📚 RAG Document Assistant

Sistema de consulta documental basado en arquitectura **RAG (Retrieval-Augmented Generation)** que permite indexar archivos PDF y realizar preguntas en lenguaje natural obteniendo respuestas fundamentadas con citas verificables por página.

## 🏗️ Arquitectura

- **Backend:** API REST desarrollada con **FastAPI** y validación estricta mediante **Pydantic**.
- **Motor Vectorial:** Extracción de texto (pypdf), generación de embeddings locales (ll-MiniLM-L6-v2) e indexación vectorial persistente con **FAISS**.
- **LLM:** Inferencia de baja latencia mediante **Groq API** (llama-3.3-70b-versatile) con control de temperatura para minimizar alucinaciones.
- **Frontend:** Interfaz de chat interactiva en **Streamlit** desacoplada del backend vía HTTP.
- **Infraestructura:** Contenerización multicontenedor con **Docker** y **Docker Compose**.

## 🚀 Ejecución rápida con Docker

1. Clona el repositorio y configura tu API Key:
   \\\ash
   cp .env.example .env
   # Añade tu GROQ_API_KEY en el archivo .env
   \\\
2. Levanta los servicios:
   \\\ash
   docker-compose up --build
   \\\
3. Accede a la interfaz web en http://localhost:8501 o a la documentación interactiva de la API en http://localhost:8000/docs.
