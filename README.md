# Litends Lab — Local RAG Application

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Zeesejo/RAG-app/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeesejo/RAG-app/actions/workflows/ci.yml)

This project is a fully local, free Retrieval Augmented Generation (RAG) app built with Python, Streamlit, LangChain, Chroma, and Ollama. Built by Litends Lab.

## Highlights
- 100% local — no cloud, no API keys
- CPU-friendly default models (works on 16GB RAM)
- Upload files or fetch URLs, per-collection storage
- Hybrid retrieval (Dense + BM25 + RRF), MMR, multi-query
- Optional local reranker (sentence-transformers cross-encoder)
- Streaming answers, citations with snippets, inline PDF previews
- Sessions (save/load/delete) and collection export/import (ZIP)
- Brandable theme and logo

Note: This app runs locally and isn’t hosted via GitHub Pages. Follow Quickstart below.

## Quickstart (Windows PowerShell)
1) Install Ollama for Windows and start it (https://ollama.com/download)
2) Create and activate venv, install deps, pull models, run:
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
- Ingest tab: upload PDFs/TXT/DOCX/MD or paste URLs (Trafilatura), then Ingest.
- Chat tab: ask questions; tune Top-K, retrieval mode, MMR, reranker, style.
- Sessions: save/load/delete from the expander.
- Export/Import: zip/unzip docs and Chroma index per collection.

## Config
- Logo: put your image at `assets/litends-logo-jul18-2nd-edition.jpg` or `assets/logo.png`.
- Theme: `.streamlit/config.toml` controls colors and fonts.

## Project structure
```
app.py
assets/
chroma_db/           # ignored by git
.docs/
docs/                # tracked; content ignored except placeholders
sessions/            # ignored by git
streamlit config
```

## License
MIT — see LICENSE.

## FAQ
- Models fail to load? Ensure Ollama is running and the models are pulled.
- Import unresolved errors for sentence-transformers? Ensure the venv is selected and dependencies installed.

## Contributing
PRs welcome. Keep it local-first and CPU-friendly.
