# app.py — Multi‑Language RAG (Streamlit + ChromaDB + SentenceTransformers)
# - Upload PDFs/DOCX/TXT (any language)
# - Cross‑lingual retrieval via multilingual embeddings
# - Translate question -> English (for generator), then answer -> target language
# - Short, factual answers; duplicate‑proof ingestion; safe reset; numeral localization

import os, sys, io, time, uuid, re
import streamlit as st

# --- Patch sqlite for Chroma on some hosts
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = __import__("pysqlite3")
except Exception:
    pass

# Core deps
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langdetect import detect
from transformers import pipeline
from typing import List, Dict, Tuple
from hashlib import sha1

# File readers
from pypdf import PdfReader
from docx import Document

# Translation (reliable for real‑world languages)
from deep_translator import GoogleTranslator

# ----------------------------
# Config
# ----------------------------
APP_TITLE = "Multi‑Language RAG"
PERSIST_DIR = "storage"
COLLECTION_NAME = "mlrag_multilingual"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODELS = {
    "flan‑t5‑small (default)": "google/flan-t5-small",
    "flan‑t5‑base (better)":  "google/flan-t5-base",
}

LANG_CHOICES = [
    "auto (use query language)",
    "English",
    "हिन्दी (Hindi)",
    "తెలుగు (Telugu)",
    "தமிழ் (Tamil)",
    "বাংলা (Bengali)",
    "मराठी (Marathi)",
    "Français (French)",
    "Deutsch (German)",
    "Español (Spanish)",
    "Português (Portuguese)",
    "Italiano (Italian)",
    "Русский (Russian)",
    "日本語 (Japanese)",
    "한국어 (Korean)",
    "中文 简体 (Chinese Simplified)",
    "中文 繁體 (Chinese Traditional)",
    "العربية (Arabic)",
    "فارسی (Persian)",
    "Türkçe (Turkish)",
    "Українська (Ukrainian)",
]

GOOGLE_LANG_CODE = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
    "বাংলা (Bengali)": "bn",
    "मराठी (Marathi)": "mr",
    "Français (French)": "fr",
    "Deutsch (German)": "de",
    "Español (Spanish)": "es",
    "Português (Portuguese)": "pt",
    "Italiano (Italian)": "it",
    "Русский (Russian)": "ru",
    "日本語 (Japanese)": "ja",
    "한국어 (Korean)": "ko",
    "中文 简体 (Chinese Simplified)": "zh-CN",
    "中文 繁體 (Chinese Traditional)": "zh-TW",
    "العربية (Arabic)": "ar",
    "فارسی (Persian)": "fa",
    "Türkçe (Turkish)": "tr",
    "Українська (Ukrainian)": "uk",
}

# ----------------------------
# Cache helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def get_client() -> chromadb.Client:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(allow_reset=False, anonymized_telemetry=False)
    )

@st.cache_resource(show_spinner=False)
def get_collection():
    client = get_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def ensure_collection():
    return get_collection()

@st.cache_resource(show_spinner=True)
def get_generator(model_id: str):
    return pipeline(
        "text2text-generation",
        model=model_id,
        max_length=384,
        truncation=True,
        num_beams=4,
        early_stopping=True
    )

@st.cache_resource(show_spinner=False)
def get_google_translator(target_label: str):
    tgt = GOOGLE_LANG_CODE.get(target_label, "en")
    try:
        return GoogleTranslator(source="auto", target=tgt)
    except Exception:
        return None

# ----------------------------
# IO helpers
# ----------------------------
def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts).strip()

def read_docx(file: io.BytesIO) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def read_txt(file: io.BytesIO) -> str:
    return file.read().decode("utf-8", errors="ignore").strip()

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def chunk_text(text: str, chunk_size=700, overlap=120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(embedder, texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()

# numeral localization for short answers
def _digit_map_for_label(target_lang_label: str):
    maps = {
        "हिन्दी (Hindi)": "०१२३४५६७८९",
        "मराठी (Marathi)": "०१२३४५६७८९",
        "తెలుగు (Telugu)": "౦౧౨౩౪౫౬౭౮౯",
        "தமிழ் (Tamil)": "௦௧௨௩௪௫௬௭௮௯",
        "বাংলা (Bengali)": "০১২৩৪৫৬৭৮৯",
    }
    return maps.get(target_lang_label, None)

def postprocess_answer(raw: str, target_lang_label: str) -> str:
    ans = (raw or "").strip()
    digit_map = _digit_map_for_label(target_lang_label)
    if digit_map:
        ans = ans.translate(str.maketrans("0123456789", digit_map))
    short = len(ans) <= 6
    numericish = len(ans) > 0 and all(ch.isdigit() or ch in " ,.-/–—" for ch in ans)
    if short or numericish:
        prefixes = {
            "हिन्दी (Hindi)": "उत्तर: ",
            "मराठी (Marathi)": "उत्तर: ",
            "తెలుగు (Telugu)": "సమాధానం: ",
            "தமிழ் (Tamil)": "பதில்: ",
            "বাংলা (Bengali)": "উত্তর: ",
            "Français (French)": "Réponse : ",
            "Deutsch (German)": "Antwort: ",
            "Español (Spanish)": "Respuesta: ",
            "Português (Portuguese)": "Resposta: ",
            "Italiano (Italian)": "Risposta: ",
            "Русский (Russian)": "Ответ: ",
            "日本語 (Japanese)": "答え：",
            "한국어 (Korean)": "답변: ",
            "中文 简体 (Chinese Simplified)": "答案：",
            "中文 繁體 (Chinese Traditional)": "答案：",
            "العربية (Arabic)": "الإجابة: ",
            "فارسی (Persian)": "پاسخ: ",
            "English": "Answer: ",
        }
        return f"{prefixes.get(target_lang_label, '')}{ans}"
    return ans

# ----------------------------
# RAG helpers
# ----------------------------
def add_docs_to_vectorstore(docs: List[Dict], embedder):
    coll = ensure_collection()
    to_add_ids, to_add_texts, to_add_metas = [], [], []
    for d in docs:
        fp = sha1((d["source"] + "".join(d["chunks"])).encode("utf-8")).hexdigest()[:12]
        stable_ids = [f"{d['source']}|{fp}|{i}" for i in range(len(d["chunks"]))]
        try:
            existing = set(coll.get(ids=stable_ids).get("ids", []))
        except Exception:
            existing = set()
        for i, ch in enumerate(d["chunks"]):
            cid = stable_ids[i]
            if cid in existing:
                continue
            to_add_ids.append(cid)
            to_add_texts.append(ch)
            to_add_metas.append({"source": d["source"], "language": d["language"]})
    if not to_add_texts:
        return 0
    embeds = embed_texts(embedder, to_add_texts)
    try:
        ensure_collection().add(
            documents=to_add_texts, metadatas=to_add_metas,
            embeddings=embeds, ids=to_add_ids
        )
    except Exception as e:
        try:
            from chromadb.errors import InvalidCollectionException
        except Exception:
            InvalidCollectionException = tuple()
        if isinstance(e, InvalidCollectionException):
            st.warning("Collection missing. Recreating and retrying indexing…")
            st.cache_resource.clear()
            _ = ensure_collection()
            ensure_collection().add(
                documents=to_add_texts, metadatas=to_add_metas,
                embeddings=embeds, ids=to_add_ids
            )
        else:
            raise
    return len(to_add_texts)

def search(query: str, k: int, embedder) -> Dict:
    coll = ensure_collection()
    try:
        total = coll.count()
        if isinstance(total, int) and total > 0:
            k = max(1, min(k, total))
    except Exception:
        pass
    q_emb = embed_texts(embedder, [query])[0]
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return {
        "documents": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
        "distances": res.get("distances", [[]])[0],
        "ids": res.get("ids", [[]])[0],
    }

def build_context(docs: List[str], max_chars=1600) -> str:
    ctx = ""
    for d in docs:
        if len(ctx) + len(d) + 4 > max_chars:
            break
        ctx += d.strip() + "\n\n"
    return ctx.strip()

def resolve_target_language(user_choice: str, query_text: str) -> Tuple[str, str]:
    if user_choice.startswith("auto"):
        qlang = detect_lang_safe(query_text)
        return (qlang, qlang)
    return (user_choice, user_choice)

def make_prompt(context: str, question_en: str, target_lang_label: str) -> str:
    # Short, factual answers; English-only for generator
    return (
        "You are a QA assistant.\n"
        "- Use ONLY the provided context.\n"
        "- If the answer isn't present, say you don't know.\n"
        "- If the answer is a person/organization/date/number, respond with JUST that short phrase.\n\n"
        f"Context:\n{context}\n\n"
        f"Question (English): {question_en}\n"
        "Answer (short):"
    )

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🌍", layout="wide")
st.title("🌍 Multi‑Language RAG")

with st.sidebar:
    st.header("⚙️ Settings")
    target_lang_choice = st.selectbox("Preferred answer language", LANG_CHOICES, index=0)
    gen_model_label = st.selectbox("Generator model", list(GEN_MODELS.keys()), index=0)
    gen_model_id = GEN_MODELS[gen_model_label]
    chunk_size = st.slider("Chunk size (chars)", 400, 1200, 700, 50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 50, 300, 120, 10)
    top_k = st.slider("Top‑K passages", 1, 8, 4, 1)
    st.divider()
    if st.button("🗑️ Reset Vector Store"):
        client = get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        st.cache_resource.clear()
        st.success("Vector store cleared. Re‑ingest documents.")

st.write("**Upload documents (PDF, DOCX, TXT) in any language. Ask questions in any language. Choose your preferred answer language in the sidebar.**")

with st.spinner("Loading models… (first run may take ~1‑2 minutes)"):
    embedder = get_embedder()
    generator = get_generator(gen_model_id)

# ---------- Ingestion ----------
st.subheader("📥 Ingest documents")
uploads = st.file_uploader("Drop your files here", type=["pdf","docx","txt"], accept_multiple_files=True)

if uploads and st.button("⚡ Process & Index"):
    docs = []
    for f in uploads:
        ext = (f.name.split(".")[-1] or "").lower()
        try:
            text = read_pdf(f) if ext=="pdf" else read_docx(f) if ext=="docx" else read_txt(f)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
            continue
        if not text.strip():
            st.warning(f"{f.name}: empty or unreadable.")
            continue
        lang = detect_lang_safe(text[:2000])
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        docs.append({"doc_id": str(uuid.uuid4())[:8], "source": f.name, "language": lang, "chunks": chunks})
    with st.spinner("Embedding & adding to vector store…"):
        n = add_docs_to_vectorstore(docs, embedder)
    st.success(f"Ingested {len(docs)} doc(s), {n} new chunk(s).")

st.divider()

# ---------- Q&A ----------
st.subheader("🔎 Ask a question")
query = st.text_input("Type your question in any language", "")

colA, colB = st.columns([1, 1])
with colA: ask_btn = st.button("💬 Retrieve & Answer")
with colB: show_ctx = st.toggle("Show retrieved passages", value=True)

if ask_btn and query.strip():
    t0 = time.time()
    hits = search(query, top_k, embedder)
    retrieved, metas = hits["documents"], hits["metadatas"]
    if not retrieved:
        st.warning("No results in vector store. Please ingest documents first.")
    else:
        # 0) Build context
        context = build_context(retrieved, max_chars=1600)

        # 1) Translate user question -> English for generator (FLAN is strongest in EN)
        qlang = detect_lang_safe(query)
        gen_question_en = query
        if qlang != "en":
            try:
                en_translator = GoogleTranslator(source="auto", target="en")
                gen_question_en = en_translator.translate(query)
            except Exception:
                gen_question_en = query  # graceful fallback

        # 2) Strong, short-answer prompt (English)
        prompt = make_prompt(context, gen_question_en, "English")

        with st.spinner("Generating answer…"):
            raw_out = generator(prompt)[0]["generated_text"]

        # 3) Translate generator output -> user's preferred language
        final_out = raw_out
        translator = get_google_translator(target_lang_choice)
        if translator and target_lang_choice != "English":
            try:
                final_out = translator.translate(raw_out)
            except Exception as e:
                st.warning(f"Translation failed; showing generator output. Details: {e}")
                final_out = raw_out

        # 4) Localize digits / wrap short numeric answers
        final_out = postprocess_answer(final_out, target_lang_choice)

        st.markdown("### ✅ Answer")
        st.write(final_out)
        st.caption(f"Latency: {time.time()-t0:.2f}s • Generator: {gen_model_label} • Embeddings: {DEFAULT_EMBED_MODEL}")

        if show_ctx:
            st.markdown("### 📚 Retrieved Passages")
            for i, (doc, meta, dist) in enumerate(zip(retrieved, metas, hits["distances"])):
                score = 1.0 - float(dist) if dist is not None else 0.0
                st.markdown(f"**{i+1}. Source:** `{meta.get('source','?')}` • **Lang:** `{meta.get('language','?')}` • **Score:** `{score:.3f}`")
                st.write(doc)
                st.markdown("---")

# Footer
st.caption("Multi‑Language RAG • Cross‑lingual embeddings + context‑aware generation • Streamlit")