# Chatbot RAG de Promtior

Este proyecto implementa un chatbot utilizando la arquitectura **RAG (Retrieval Augmented Generation)** para responder preguntas sobre la información de la empresa **Promtior**.  
La información se extrae únicamente de fuentes provistas por nosotros (páginas web de Promtior y PDF de presentación).

## Tecnologías usadas

- Python 3.12
- FastAPI para la API
- LangChain para la arquitectura RAG
- Ollama LLaMA 3 para embeddings y LLM
- Chroma para vectorstore
- Document loaders: WebBaseLoader y PyPDFLoader
- RecursiveCharacterTextSplitter para dividir textos en chunks

## Cómo usar

1. Ingresar al endpoint `/ask` enviando un POST con JSON:
```json
{
  "question": "¿Qué servicios ofrece Promtior?"
}
