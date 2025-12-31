import os
from pathlib import Path
import io

import gradio as gr
from PIL import Image
import fitz  # PyMuPDF

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# ================== CONFIG ==================
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

PDF_PATH = r"C:\Users\Karishma\colakin\colakinfinal2\sample.pdf"
FAISS_DIR = r"C:\Users\Karishma\colakin\colakinfinal2\faiss_index"
EMBED_MODEL = "hkunlp/instructor-xl"
TOP_K = 4
MAX_RETURN_CHUNKS = 4  # Maximum chunks to provide context to Gemini

# Ensure FAISS directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

# ================== PDF → Images ==================
def pdf_to_images(pdf_path: str):
    doc = fitz.open(pdf_path)
    imgs = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imgs.append(img)
    return imgs

# ================== FAISS DB ==================
def build_or_load_db(pdf_path, faiss_dir):
    # Load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Split pages into smaller chunks for precise retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # Keep track of original page numbers
    for chunk in chunks:
        chunk.metadata["page"] = chunk.metadata.get("page", 0)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Load or create FAISS index
    if Path(faiss_dir).exists():
        try:
            db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        except:
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(faiss_dir)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(faiss_dir)

    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    return db, images

# ================== INIT ==================
db, pdf_images = build_or_load_db(PDF_PATH, FAISS_DIR)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY
)

# ================== QUERY HANDLER ==================
def answer_with_sources_agno(query: str):
    q = query.strip().lower()
    if not q:
        return "Please ask a question.", [], []

    # Greeting handling
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon",
                         "good evening", "how are you", "what's up"]
    if any(word in q for word in greeting_keywords):
        return "Hello! I’m your Agno assistant. Ask me anything from the PDF.", [], []

    # FAISS chunk search
    chunks = db.similarity_search(query, k=TOP_K)
    if not chunks:
        return "I’m sorry. I can only answer questions from the provided PDF.", [], []

    # Take top chunks for context
    top_chunks = chunks[:MAX_RETURN_CHUNKS]

    # Prepare context for LLM
    context = "\n\n".join([c.page_content for c in top_chunks])
    prompt = (
        "You are an intelligent Agno assistant. Answer strictly using ONLY the provided PDF content. "
        "Do NOT invent answers.\n\n"
        "If the answer is not available, reply: 'I cannot find the answer in the provided document.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer clearly and concisely."
    )

    # Call Gemini
    res = llm.invoke(prompt)
    answer = getattr(res, "content", str(res))

    # Map relevant chunks to page images
    total_pages = len(pdf_images)
    ordered_imgs, ordered_pages, seen = [], [], set()
    for c in top_chunks:
        p = c.metadata.get("page", 0)
        if isinstance(p, int) and 0 <= p < total_pages and p not in seen:
            seen.add(p)
            ordered_imgs.append(pdf_images[p])
            ordered_pages.append(p + 1)

    if ordered_pages:
        answer += f"\n\n(Pages: {', '.join(map(str, ordered_pages))})"

    return answer, ordered_imgs, ordered_pages

# ================== AGNO GRADIO UI ==================
with gr.Blocks() as demo:
    gr.Markdown("## ✅ Agno PDF Assistant")

    bot = gr.Chatbot()
    gallery = gr.Gallery(label="Relevant Pages")
    inp = gr.Textbox(label="Ask a question")

    def respond(q, history):
        ans, imgs, pages = answer_with_sources_agno(q)
        history.append((q, ans))
        return history, imgs

    inp.submit(respond, [inp, bot], [bot, gallery])

demo.launch()
