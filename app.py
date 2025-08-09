import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import BM25Retriever
from typing import List, Dict
import os
import shutil
import re
from html import escape
import io
import fitz  # PyMuPDF
import trafilatura
import json
import zipfile
from datetime import datetime
from pathlib import Path
import hashlib

THINKING = "Thinking..."

# Branding (use provided logo with fallback)
logo_candidates = [
    os.path.join("assets", "litends-logo-jul18-2nd-edition.jpg"),
    os.path.join("assets", "logo.png"),
]
logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)
page_icon = logo_path if logo_path else "🧠"

st.set_page_config(page_title="Litends Lab — Local RAG", page_icon=page_icon, layout="wide")

# Brand CSS
st.markdown(
    """
    <style>
      /* Buttons */
      .stButton > button {
        background-color: #8B5CF6; /* primary */
        border: 1px solid #7C3AED;
        color: #ffffff;
        border-radius: 8px;
      }
      .stButton > button:hover { background-color: #7C3AED; border-color:#6D28D9; }
      .stButton > button:focus { box-shadow: 0 0 0 0.2rem rgba(139,92,246,0.35); outline: none; }

      /* Inputs */
      .stTextInput > div > div > input,
      .stTextArea textarea {
        border: 1px solid #1F2937;
        border-radius: 8px;
        background-color: #0E1117;
        color: #E5E7EB;
      }
      .stTextInput > div > div > input:focus,
      .stTextArea textarea:focus { border-color: #8B5CF6; box-shadow: 0 0 0 1px #8B5CF6; }

      /* Expanders */
      details > summary {
        background-color: #111827;
        color: #E5E7EB;
        border: 1px solid #1F2937;
        border-radius: 6px;
        padding: 8px 12px;
      }
      details[open] > summary { background-color: #0F172A; }
      details > div { background: #0E1117; border: 1px solid #1F2937; border-top: none; border-radius: 0 0 6px 6px; }

      /* Links */
      a { color: #06B6D4; }
      a:hover { color: #22D3EE; text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Branded header
if logo_path:
    col_logo, col_title = st.columns([1, 7])
    with col_logo:
        st.image(logo_path, width=64)
    with col_title:
        st.markdown("# Litends Lab — Local RAG Chat")
        st.caption("Ollama + Chroma + LangChain (100% local)")
else:
    st.markdown("# Litends Lab — Local RAG Chat")
    st.caption("Ollama + Chroma + LangChain (100% local)")

# Sidebar branding + nav
with st.sidebar:
    if logo_path:
        st.image(logo_path, width=120)
    st.markdown("### Built by Litends Lab")
    nav = st.radio("Navigate", ["Home", "Ingest", "Chat", "About"], index=0)
    st.divider()
    st.caption("Workspace & Retrieval Settings")
    collection = st.text_input("Collection name", value=st.session_state.get("collection", "default"))
    st.session_state["collection"] = collection.strip() or "default"
    retrieval_mode = st.selectbox("Retrieval mode", ["dense", "bm25", "hybrid"], index=0)
    multi_query = st.checkbox("Multi‑query expansion", value=False, help="Generate variations of your question and fuse results.")
    stream_answer = st.checkbox("Stream answer", value=False)
    # Reranker
    rerank_answers = st.checkbox(
        "Rerank answers (cross-encoder)", value=False,
        help="Reorder retrieved chunks using a local cross-encoder for better answer quality."
    )
    reranker_model = st.text_input("Reranker model", value=st.session_state.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    # MMR
    use_mmr = st.checkbox("Use MMR (diversified dense retrieval)", value=False)
    mmr_lambda = st.slider("MMR diversity (λ)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Higher = more diversity")
    # Confidence gating
    min_sources = st.slider("Min sources to answer", min_value=1, max_value=10, value=2, step=1, help="If fewer sources retrieved, answer with uncertainty.")
    # Style
    answer_style = st.selectbox("Answer style", ["Default", "Concise", "Bullet points", "Step-by-step", "Detailed"], index=0)
    top_k_val = st.slider("Top-K results", min_value=1, max_value=10, value=4, step=1)
    chunk_size_val = st.slider("Chunk size", min_value=256, max_value=1200, value=512, step=32)
    chunk_overlap_val = st.slider("Chunk overlap", min_value=0, max_value=256, value=64, step=16)
    model_name = st.text_input("Ollama model", value=st.session_state.get("model", "llama3.1:8b"))
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    st.session_state.update({
        "retrieval_mode": retrieval_mode,
        "multi_query": multi_query,
        "stream_answer": stream_answer,
        "top_k": top_k_val,
        "chunk_size": chunk_size_val,
        "chunk_overlap": chunk_overlap_val,
        "model": model_name,
        "temperature": float(temperature),
        "rerank": rerank_answers,
        "reranker_model": reranker_model,
        "use_mmr": use_mmr,
        "mmr_lambda": float(mmr_lambda),
        "min_sources": int(min_sources),
        "answer_style": answer_style,
    })

# Paths per collection
collection = st.session_state.get("collection", "default")
docs_dir = os.path.join("docs", collection)
chroma_dir = os.path.join("chroma_db", collection)
os.makedirs(docs_dir, exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

# Loaders

def load_docs(folder):
    docs = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        ext = fname.lower().split(".")[-1]
        try:
            if ext == "pdf":
                docs.extend(PyPDFLoader(path).load())
            elif ext in ("txt", "md"):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif ext == "docx":
                docs.extend(Docx2txtLoader(path).load())
        except Exception as e:
            st.warning(f"Failed to load {fname}: {e}")
    return docs

# Utils

def make_snippet(text: str, query: str, max_len: int = 400) -> str:
    if not text:
        return ""
    q = (query or "").strip()
    pos = text.lower().find(q.lower()) if q else -1
    if pos == -1:
        start = 0
    else:
        start = max(0, pos - 120)
    end = min(len(text), start + max_len)
    snippet = text[start:end]
    escaped = escape(snippet)
    if q:
        pattern = re.compile(re.escape(escape(q)), re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return ("…" if start > 0 else "") + escaped + ("…" if end < len(text) else "")

@st.cache_resource(show_spinner=False)
def build_bm25_index(docs_path: str, chunk_size: int, chunk_overlap: int) -> BM25Retriever:
    raw_docs = []
    for fname in os.listdir(docs_path):
        path = os.path.join(docs_path, fname)
        ext = fname.lower().split(".")[-1]
        try:
            if ext == "pdf":
                raw_docs.extend(PyPDFLoader(path).load())
            elif ext in ("txt", "md"):
                raw_docs.extend(TextLoader(path, encoding="utf-8").load())
            elif ext == "docx":
                raw_docs.extend(Docx2txtLoader(path).load())
        except Exception:
            continue
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(raw_docs) if raw_docs else []
    return BM25Retriever.from_documents(splits) if splits else BM25Retriever(docs=[])

@st.cache_resource(show_spinner=False)
def bm25_cache_key(docs_path: str, chunk_size: int, chunk_overlap: int) -> str:
    h = hashlib.sha256()
    h.update((docs_path + "|" + str(chunk_size) + "|" + str(chunk_overlap)).encode("utf-8"))
    # incorporate folder mtimes
    try:
        mtimes = [str(os.path.getmtime(os.path.join(docs_path, f))) for f in os.listdir(docs_path)]
    except Exception:
        mtimes = []
    h.update("|".join(sorted(mtimes)).encode("utf-8"))
    return h.hexdigest()

@st.cache_resource(show_spinner=False)
def build_bm25_index_cached(docs_path: str, chunk_size: int, chunk_overlap: int) -> BM25Retriever:
    _ = bm25_cache_key(docs_path, chunk_size, chunk_overlap)
    return build_bm25_index(docs_path, chunk_size, chunk_overlap)

def rrf_fuse(lists: List[List], k: int = 60, limit: int = 8):
    scores: Dict[str, float] = {}
    keeper: Dict[str, any] = {}
    for results in lists:
        for rank, d in enumerate(results):
            key = f"{d.metadata.get('source')}#{d.metadata.get('page', -1)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in keeper:
                keeper[key] = d
    fused = sorted(keeper.values(), key=lambda d: scores[f"{d.metadata.get('source')}#{d.metadata.get('page', -1)}"], reverse=True)
    return fused[:limit]

# Reranker (optional, CPU-friendly cross-encoder)
@st.cache_resource(show_spinner=False)
def get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device='cpu')


def rerank_docs(query: str, docs: List, top_n: int, model_name: str) -> List:
    if not docs:
        return []
    model = get_reranker(model_name)
    pairs = [(query, getattr(d, 'page_content', '') or '') for d in docs]
    scores = model.predict(pairs)
    try:
        scores = scores.tolist()
    except Exception:
        pass
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:max(1, top_n)]]

# Views

def render_home():
    st.subheader("Welcome to Litends Lab Local RAG")
    st.write("Chat with your documents 100% locally using Ollama + Chroma + LangChain.")
    st.markdown("- Go to Ingest to upload/prepare documents\n- Then Chat to ask questions and get cited answers")
    with st.expander("How it works"):
        st.write("Uploads are chunked and embedded with nomic-embed-text, stored in Chroma, and queried with Llama 3.1 8B via Ollama.")


def render_ingest():
    st.subheader(f"Ingest Documents — Collection: {collection}")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX, MD)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(docs_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to {docs_dir}/")

    # URL ingestion
    with st.expander("Add web pages by URL"):
        urls_text = st.text_area("URLs (one per line)", placeholder="https://example.com/page\nhttps://another/page")
        if st.button("Fetch & Save URLs") and urls_text.strip():
            added = 0
            for raw in urls_text.splitlines():
                url = raw.strip()
                if not url:
                    continue
                try:
                    downloaded = trafilatura.fetch_url(url)
                    text = trafilatura.extract(downloaded) if downloaded else None
                    if text:
                        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", url)[:60]
                        out_path = os.path.join(docs_dir, f"url-{safe_name}.txt")
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        added += 1
                except Exception as e:
                    st.warning(f"Failed {url}: {e}")
            st.success(f"Saved {added} page(s) into {docs_dir}/")

    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Ingest Documents", use_container_width=True):
            with st.spinner(THINKING):
                raw_docs = []
                for fname in os.listdir(docs_dir):
                    path = os.path.join(docs_dir, fname)
                    ext = fname.lower().split(".")[-1]
                    try:
                        if ext == "pdf":
                            raw_docs.extend(PyPDFLoader(path).load())
                        elif ext in ("txt", "md"):
                            raw_docs.extend(TextLoader(path, encoding="utf-8").load())
                        elif ext == "docx":
                            raw_docs.extend(Docx2txtLoader(path).load())
                    except Exception as e:
                        st.warning(f"Failed to load {fname}: {e}")
                if not raw_docs:
                    st.warning("No documents found. Upload or fetch URLs first.")
                    return
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=st.session_state.get("chunk_size", 512),
                    chunk_overlap=st.session_state.get("chunk_overlap", 64),
                )
                splits = splitter.split_documents(raw_docs)
                embed = OllamaEmbeddings(model="nomic-embed-text")
                Chroma.from_documents(splits, embed, persist_directory=chroma_dir)
            st.success(f"Ingested {len(splits)} chunks into {chroma_dir}.")
    with cols[1]:
        if st.button("Re-index (clear & ingest)", type="secondary", use_container_width=True):
            with st.spinner("Re-indexing..."):
                try:
                    if os.path.exists(chroma_dir):
                        shutil.rmtree(chroma_dir, ignore_errors=True)
                except Exception as e:
                    st.error(f"Failed to clear {chroma_dir}: {e}")
                    return
                # Trigger ingest (same as above)
                raw_docs = []
                for fname in os.listdir(docs_dir):
                    path = os.path.join(docs_dir, fname)
                    ext = fname.lower().split(".")[-1]
                    try:
                        if ext == "pdf":
                            raw_docs.extend(PyPDFLoader(path).load())
                        elif ext in ("txt", "md"):
                            raw_docs.extend(TextLoader(path, encoding="utf-8").load())
                        elif ext == "docx":
                            raw_docs.extend(Docx2txtLoader(path).load())
                    except Exception:
                        continue
                if not raw_docs:
                    st.info("No documents to index.")
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=st.session_state.get("chunk_size", 512),
                        chunk_overlap=st.session_state.get("chunk_overlap", 64),
                    )
                    splits = splitter.split_documents(raw_docs)
                    embed = OllamaEmbeddings(model="nomic-embed-text")
                    Chroma.from_documents(splits, embed, persist_directory=chroma_dir)
                    st.success(f"Re-indexed {len(splits)} chunks into {chroma_dir}.")

    # NEW: Export/Import tools
    with st.expander("Export / Import collection"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export collection as ZIP", use_container_width=True):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    # include docs/<collection>
                    for root, _, files in os.walk(docs_dir):
                        for fn in files:
                            full = os.path.join(root, fn)
                            arc = os.path.relpath(full, start=".")
                            z.write(full, arcname=arc)
                    # include chroma_db/<collection>
                    if os.path.exists(chroma_dir):
                        for root, _, files in os.walk(chroma_dir):
                            for fn in files:
                                full = os.path.join(root, fn)
                                arc = os.path.relpath(full, start=".")
                                z.write(full, arcname=arc)
                st.download_button(
                    "Download ZIP",
                    data=buf.getvalue(),
                    file_name=f"collection-{collection}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        with c2:
            zip_upload = st.file_uploader("Import collection from ZIP", type=["zip"], accept_multiple_files=False)
            if zip_upload is not None and st.button("Import ZIP", use_container_width=True):
                try:
                    base = Path(".").resolve()
                    with zipfile.ZipFile(zip_upload) as z:
                        members = [m for m in z.namelist() if m.startswith(("docs/", "chroma_db/"))]
                        for m in members:
                            dest = (base / m).resolve()
                            if not str(dest).startswith(str(base)):
                                continue
                            if m.endswith("/"):
                                os.makedirs(dest, exist_ok=True)
                            else:
                                os.makedirs(dest.parent, exist_ok=True)
                                with z.open(m) as src, open(dest, "wb") as out:
                                    out.write(src.read())
                    st.success("Imported collection files. Re-index if needed.")
                except Exception as e:
                    st.error(f"Import failed: {e}")

def render_chat():
    st.subheader(f"Chat with your Documents — Collection: {collection}")
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        st.warning("No index found for this collection. Go to Ingest and create the vector DB first.")
        return

    # Chat history state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Session controls
    os.makedirs("sessions", exist_ok=True)
    with st.expander("Session controls"):
        default_session_name = st.session_state.get("session_name", f"{collection}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        session_name = st.text_input("Session name", value=default_session_name, key="session_name")
        s1, s2, s3 = st.columns([1,1,2])
        with s1:
            if st.button("Save session", use_container_width=True):
                safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_name)[:80]
                data = {
                    "name": safe,
                    "collection": collection,
                    "history": st.session_state.history,
                    "settings": {
                        "retrieval_mode": st.session_state.get("retrieval_mode"),
                        "multi_query": st.session_state.get("multi_query"),
                        "stream_answer": st.session_state.get("stream_answer"),
                        "top_k": st.session_state.get("top_k"),
                        "chunk_size": st.session_state.get("chunk_size"),
                        "chunk_overlap": st.session_state.get("chunk_overlap"),
                        "model": st.session_state.get("model"),
                        "temperature": st.session_state.get("temperature"),
                        "rerank": st.session_state.get("rerank"),
                        "reranker_model": st.session_state.get("reranker_model"),
                    },
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
                with open(os.path.join("sessions", f"{safe}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                st.success("Session saved.")
        with s2:
            # load selector
            files = [f for f in os.listdir("sessions") if f.endswith(".json")]
            choice = st.selectbox("Saved sessions", files, index=0 if files else None, placeholder="Select a session")
            if st.button("Load", use_container_width=True, disabled=not bool(files)):
                try:
                    with open(os.path.join("sessions", choice), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    st.session_state.history = data.get("history", [])
                    # restore settings
                    for k, v in data.get("settings", {}).items():
                        st.session_state[k] = v
                    st.session_state["collection"] = data.get("collection", st.session_state.get("collection", "default"))
                    st.success("Session loaded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")
        with s3:
            if st.button("Delete selected", use_container_width=True, disabled=not bool(files)):
                try:
                    os.remove(os.path.join("sessions", choice))
                    st.success("Deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

    # Show existing transcript
    if st.session_state.history:
        with st.expander("Chat transcript", expanded=False):
            for i, turn in enumerate(st.session_state.history, 1):
                st.markdown(f"**Q{i}:** {turn['q']}")
                st.markdown(f"**A{i}:** {turn['a']}")
                if turn.get("sources"):
                    st.markdown("Sources:")
                    for s in turn["sources"]:
                        st.markdown(f"- {s}")

    query = st.text_input("Ask a question:")

    cols = st.columns([1,1,6])
    with cols[0]:
        ask_clicked = st.button("Ask", use_container_width=True)
    with cols[1]:
        reset_clicked = st.button("Reset Chat", type="secondary", use_container_width=True)

    if reset_clicked:
        st.session_state.history = []
        st.rerun()

    if ask_clicked and query:
        embed = OllamaEmbeddings(model="nomic-embed-text")
        llm = OllamaLLM(model=st.session_state.get("model", "llama3.1:8b"), temperature=st.session_state.get("temperature", 0.1))
        from langchain.chains import RetrievalQA

        mode = st.session_state.get("retrieval_mode", "dense")
        top_k = st.session_state.get("top_k", 4)
        do_multi = st.session_state.get("multi_query", False)
        do_stream = st.session_state.get("stream_answer", False)
        do_rerank = st.session_state.get("rerank", False)
        rerank_model_name = st.session_state.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        use_mmr = st.session_state.get("use_mmr", False)
        mmr_lambda = float(st.session_state.get("mmr_lambda", 0.5))
        min_sources = int(st.session_state.get("min_sources", 2))
        answer_style = st.session_state.get("answer_style", "Default")
        sources = []
        result_text = ""
        source_docs = []

        def style_suffix(style: str) -> str:
            return {
                "Default": "",
                "Concise": " Respond concisely in 2-3 sentences.",
                "Bullet points": " Respond as bullet points.",
                "Step-by-step": " Explain step-by-step.",
                "Detailed": " Provide a thorough, detailed explanation.",
            }.get(style, "")

        if mode == "dense" and not do_multi:
            vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embed)
            if use_mmr:
                retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(10, top_k*3), "lambda_mult": mmr_lambda})
            else:
                retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
            qa = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
            with st.spinner(THINKING):
                result = qa.invoke({"query": query})
            result_text = result["result"]
            source_docs = result["source_documents"]
        elif mode == "bm25" and not do_multi:
            bm25 = build_bm25_index_cached(docs_dir, st.session_state.get("chunk_size", 512), st.session_state.get("chunk_overlap", 64))
            bm25.k = top_k  # type: ignore[attr-defined]
            qa = RetrievalQA.from_chain_type(llm, retriever=bm25, return_source_documents=True)
            with st.spinner(THINKING):
                result = qa.invoke({"query": query})
            result_text = result["result"]
            source_docs = result["source_documents"]
        else:
            vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embed)
            dense_docs = []
            if do_multi:
                q_prompt = (
                    "Generate 3 alternative search queries for the following question, one per line, no numbering, concise.\n"
                    f"Question: {query}"
                )
                try:
                    variants = llm.invoke(q_prompt)
                    alt_queries = [q.strip("- • \t").strip() for q in variants.splitlines() if q.strip()]
                    alt_queries = alt_queries[:3] or [query]
                except Exception:
                    alt_queries = [query]
                for q in alt_queries:
                    if use_mmr:
                        dense_docs.extend(vectordb.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(10, top_k*3), "lambda_mult": mmr_lambda}).get_relevant_documents(q))
                    else:
                        dense_docs.extend(vectordb.as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(q))
            else:
                if use_mmr:
                    dense_docs = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": max(10, top_k*3), "lambda_mult": mmr_lambda}).get_relevant_documents(query)
                else:
                    dense_docs = vectordb.as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(query)

            bm25_docs = []
            if mode in ("bm25", "hybrid"):
                bm25 = build_bm25_index_cached(docs_dir, st.session_state.get("chunk_size", 512), st.session_state.get("chunk_overlap", 64))
                bm25_docs = bm25.get_relevant_documents(query)

            fused = rrf_fuse([dense_docs, bm25_docs], limit=top_k)
            if do_rerank:
                fused = rerank_docs(query, fused, top_k, rerank_model_name)
            context = "\n\n".join(d.page_content for d in fused)
            prompt = (
                "You are a helpful assistant. Answer using the given context. If unsure, say you don't know." + style_suffix(answer_style) + "\n"
                "Cite sources as [file:page] when relevant.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            with st.spinner(THINKING):
                if do_stream:
                    def gen():
                        for chunk in llm.stream(prompt):
                            yield chunk
                    result_text = st.write_stream(gen)
                else:
                    result_text = llm.invoke(prompt)
            source_docs = fused

        # Confidence gating
        if len(source_docs) < min_sources:
            result_text = ("I'm not fully confident based on the retrieved documents. "
                           "Consider adding more relevant files or rephrasing the question.")

        # If rerank was requested but we used a chain path, re-answer using reranked context
        if do_rerank and source_docs and mode in ("dense", "bm25") and not do_multi:
            try:
                reranked = rerank_docs(query, source_docs, top_k, rerank_model_name)
                context = "\n\n".join(d.page_content for d in reranked)
                prompt = (
                    "You are a helpful assistant. Answer using the given context. If unsure, say you don't know." + style_suffix(answer_style) + "\n"
                    "Cite sources as [file:page] when relevant.\n\n"
                    f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
                )
                with st.spinner("Reranking and answering..."):
                    if do_stream:
                        def gen2():
                            for chunk in llm.stream(prompt):
                                yield chunk
                        result_text = st.write_stream(gen2)
                    else:
                        result_text = llm.invoke(prompt)
                source_docs = reranked
            except Exception as e:
                st.info(f"Rerank skipped: {e}")

        st.markdown(f"**Answer:** {result_text}")
        with st.expander("Show sources"):
            for doc in source_docs:
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")
                label = f"{os.path.basename(str(src))}{f' — p.{page+1}' if isinstance(page, int) else ''}"
                sources.append(label)
                snippet_html = make_snippet(doc.page_content or "", query)
                st.markdown(f"**{escape(label)}**", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.9rem; line-height:1.4;'>{snippet_html}</div>", unsafe_allow_html=True)

                # PDF page preview
                if isinstance(page, int) and str(src).lower().endswith(".pdf") and os.path.exists(str(src)):
                    try:
                        pdf = fitz.open(str(src))
                        if 0 <= page < pdf.page_count:
                            pix = pdf[page].get_pixmap(dpi=120)
                            img_bytes = pix.tobytes("png")
                            st.image(io.BytesIO(img_bytes), caption=label, use_column_width=True)
                        pdf.close()
                    except Exception as e:
                        st.caption(f"(Preview unavailable: {e})")

                st.markdown("---")

        # Save turn
        st.session_state.history.append({"q": query, "a": result_text, "sources": sources})

    # Download transcript
    if st.session_state.history:
        lines = ["# Chat Transcript"]
        for i, t in enumerate(st.session_state.history, 1):
            lines.append(f"\n## Q{i}\n{t['q']}")
            lines.append(f"\n### A{i}\n{t['a']}")
            if t.get("sources"):
                lines.append("\nSources:")
                lines.extend([f"- {s}" for s in t["sources"]])
        md_text = "\n".join(lines)
        st.download_button(
            label="Download transcript (Markdown)",
            data=md_text.encode("utf-8"),
            file_name="litends_rag_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

def render_about():
    st.subheader("About Litends Lab")
    st.write(
        "Litends Lab builds local-first AI tools that are private, fast, and simple. "
        "This RAG app is designed to run fully on your machine with 16GB RAM."
    )
    st.markdown("""
    **Tech stack:** Streamlit • LangChain • Chroma • Ollama
    
    - 100% local, free and private
    - CPU-friendly defaults (Llama 3.1 8B, quantized)
    - Easy ingestion and cited answers
    """)

# Router
if nav == "Home":
    render_home()
elif nav == "Ingest":
    render_ingest()
elif nav == "Chat":
    render_chat()
else:
    render_about()

# Footer
st.markdown(
    """
    <style>
    .litends-footer {position: fixed; left: 0; bottom: 0; width: 100%; text-align: center; padding: 8px 0; color: #808080; background: transparent;}
    </style>
    <div class='litends-footer'>Built locally by <b>Litends Lab</b></div>
    """,
    unsafe_allow_html=True,
)
