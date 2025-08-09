<p align="center">
  <img src="../assets/litends-logo-jul18-2nd-edition.jpg" alt="Litends Lab" width="160" />
</p>

# Litends Lab — Local RAG App (Overview)

Created by: Zeeshan Modi

Repository: https://github.com/Zeesejo/RAG-app

## Summary
A fully local, free Retrieval Augmented Generation (RAG) application that lets you chat with your own documents on a Windows laptop. No cloud, no API keys. CPU‑friendly and designed to run on 16GB RAM.

## Key Features
- 100% local: Streamlit UI, LangChain orchestration, Chroma vector DB, Ollama LLM + embeddings
- Hybrid retrieval: Dense + BM25 with Reciprocal Rank Fusion (RRF)
- Options: MMR (diversified search), multi‑query expansion, optional cross‑encoder reranker
- URL ingestion (Trafilatura), inline PDF page previews (PyMuPDF), cited snippets
- Sessions: save/load/delete; export/import entire collections as ZIP
- Brandable theme and logo

## Architecture
- LLM: Ollama (llama3.1:8b) — quantized, CPU‑friendly
- Embeddings: Ollama (nomic‑embed‑text)
- Vector store: Chroma (persisted per collection)
- Orchestration: LangChain (+ langchain‑chroma, langchain‑ollama)
- UI: Streamlit

All components run locally; no external services are required.

## Requirements
- Windows 10/11, PowerShell
- Python 3.10+
- Ollama for Windows (running locally)
- ~16GB RAM recommended

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text
streamlit run app.py
```
Open http://localhost:8501

## Usage
1) Ingest tab
- Upload PDFs/TXT/DOCX/MD or paste URLs; click Ingest to index.
- Manage collections per project/topic.
- Export/Import collection to move between machines.

2) Chat tab
- Ask questions; see cited answers with snippets and PDF previews.
- Controls: Top‑K, retrieval mode (dense/bm25/hybrid), MMR, multi‑query, reranker, answer style, streaming.
- Save/load/delete sessions from the expander.

## Advanced Options
- Hybrid search with RRF for better recall.
- MMR (λ control) to balance relevance/diversity.
- Multi‑query expansion to broaden recall.
- Optional cross‑encoder reranker (sentence‑transformers) to improve ranking.

## Branding & Theming
- Logo: place your image in `assets/` (app prefers `litends-logo-jul18-2nd-edition.jpg` or `logo.png`).
- Theme: adjust `.streamlit/config.toml` (colors, font).

## Troubleshooting
- Ensure Ollama is running and models are pulled.
- If imports complain, confirm the virtual environment is active and dependencies are installed.

## License
MIT License. See LICENSE.
