import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import threading
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse




os.environ["USER_AGENT"] = "promtiortest"

app = FastAPI(title="Promtior RAG Chatbot", version="1.0")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la API key. Poné OPENAI_API_KEY en tu entorno")

# --- Request model ---
class QuestionRequest(BaseModel):
    question: str

# --- Global state ---
vectorstore_ready = False
rag_chain = None

# --- Function to load docs, split, create vectorstore and chain ---
def initialize_vectorstore():
    global vectorstore_ready, rag_chain

    print("Loading documents...")

    # --- Load PDFs ---
    pdf_loader = PyPDFLoader("infopdf.pdf")
    pdf_docs = pdf_loader.load()
    print(f"PDF loaded: {len(pdf_docs)} pages")

    # --- Load web pages ---
    web_loader = WebBaseLoader([
        "https://www.promtior.ai",
        "https://www.promtior.ai/use-case",
        "https://www.promtior.ai/service",
        "https://careers.promtior.ai/",
        "https://careers.promtior.ai/connect",
        "https://www.promtior.ai/contacto",
        "https://www.promtior.ai/blog",
        "https://careers.promtior.ai/#jobs",
        "https://www.promtior.ai/blog/categories/bionic-organizations",
        "https://www.promtior.ai/blog/categories/impact-stories",
        "https://www.promtior.ai/blog/categories/mindcraft-ai",
        "https://www.promtior.ai/blog/categories/news",
        "https://www.promtior.ai/blog/categories/tech-talks-1"
    ])
    web_docs = web_loader.load()
    print(f"Web loaded: {len(web_docs)} documents")

    # --- Combine PDFs + web ---
    all_docs = pdf_docs + web_docs
    print(f"Total raw documents: {len(all_docs)}")

    # --- Split into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Total chunks after split: {len(splits)}")

    # --- Create embeddings and vectorstore ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    CHROMA_DIR = "./chroma_db"

    if os.path.exists(CHROMA_DIR):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        print("Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

    # --- Retriever ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # --- LLM & prompt ---
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
    prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions ONLY using the context provided.
You are an assistant answering questions about Promtior which is a company.
Answer in a natural, friendly, and professional way.
Do not return bullet lists unless the user asks for them.
If the user message is a greeting, thanks, or small talk (like "thanks", "ok", "great", "hello"), respond politely and naturally without using the context.
Only use the context when the user asks an actual information question about Promtior.
Do NOT say "I don't know" to greetings or thanks.
Write like you are explaining to a client.
Do NOT generate any information that is not explicitly or clearly inferable from the context.
When the answer mentions the company it means promtior.
Do NOT make assumptions, guesses, or use prior knowledge outside of the provided documents.

The context may be in English or Spanish. The user question may be in a different language.
You may internally translate the context or question to understand it, but always answer based on the context.

Instructions:
- Only use the content from the documents provided.
- If the context does not contain enough information to answer, respond with: "I don't know."
- Do NOT invent dates, numbers, or services that are not in the documents.
- Treat synonyms or translations as equivalent 
- Answer concisely and clearly.

Context:{context}
Question:{question}
Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Build RAG chain ---
    global rag_chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    vectorstore_ready = True
    print("Vectorstore ready. API is live!")

# --- Start loading in background ---
threading.Thread(target=initialize_vectorstore, daemon=True).start()

# --- API endpoint ---
@app.post("/ask")
def ask_question(q: QuestionRequest):
    if not vectorstore_ready:
        raise HTTPException(
            status_code=503,
            detail="Vectorstore is still loading. Please try again in a few seconds."
        )

    answer = rag_chain.invoke(q.question)
    return {"question": q.question, "answer": answer}

# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "Promtior RAG Chatbot API is running. Use /ask to ask questions."}
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/chat")
def chat():
    return FileResponse("static/index.html")
