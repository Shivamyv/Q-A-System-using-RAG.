# Intelligent Document Q&A System using RAG

Ask questions about PDF documents using retrieval-augmented generation (RAG). The app extracts text, chunks it, embeds chunks, retrieves relevant passages with FAISS, and sends those passages as context to an LLM.

## Setup
Requires an OpenAI API key for answer generation. Keep keys in `.env` and never commit them.
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your API key
streamlit run app.py
```

Upload one or more text-based PDFs, click **Index documents**, then ask a question. Scanned PDFs may need OCR, which this starter does not implement.

## Design and limitations
- PDF extraction with PyMuPDF
- Chunking and embeddings through LangChain integrations
- FAISS similarity search
- Retrieved context is passed to the chat model
- Answers include retrieved source filenames and page numbers

The system can still produce incorrect answers. Verify important information against the cited source passages. Do not upload confidential documents unless you have permission and understand the provider's data-handling terms.
