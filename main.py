import os
os.environ["USER_AGENT"] = "promtiortest"

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



pdf_loader = PyPDFLoader("infopdf.pdf")
pdf_docs = pdf_loader.load()
print(f"PDF loaded: {len(pdf_docs)} pages")



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
all_docs = pdf_docs + web_docs
print(f"Total raw documents: {len(all_docs)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)
splits = text_splitter.split_documents(all_docs)
print(f"Total chunks after split: {len(splits)}")

embeddings = OllamaEmbeddings(model="llama3")

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

# Si querés regenerar todo desde cero borrá la carpeta chroma_db antes de correr

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


llm = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template("""
You are an assistant answering questions about Promtior.

The context may be in English or Spanish.
The user question may be in a different language than the context.
You may translate the question or the context internally to find the answer.

Treat synonyms and translations as equivalent, for example:
- "fundada" = "founded"
- "años" = "years"
- "empleados" = "employees"

First, check the context for exact or inferred answers.
Then, answer the question concisely.

Use ONLY the provided context to answer.
If the context does not contain enough information, say "I don't know".

Context:{context}
Question:{question}
Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

questions = [
    "What services does Promtior offer?",
    "When was the company founded?"
]

for q in questions:
    print(f"\nQuestion: {q}")
    answer = rag_chain.invoke(q)
    print(f"Answer: {answer}")