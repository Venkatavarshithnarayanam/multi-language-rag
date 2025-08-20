# Multi‑Language RAG System

> Problem statement
>
> Build a Retrieval-Augmented Generation (RAG) system that can retrieve information from documents in multiple languages and provide answers in the user's preferred language, handling translation and cultural context appropriately.

---


This repository contains a complete Multi‑Language RAG System designed to:

* Ingest documents in multiple languages (text, PDF, optionally audio/image OCR).
* Create multilingual embeddings and index them in a vector store.
* Retrieve relevant context across languages (cross-lingual retrieval).
* Translate, preserve cultural context, and generate answers in the user’s preferred language.
* Deploy a demo (Streamlit / Gradio / Hugging Face Spaces).

This project is suitable for a domain-focused RAG (e.g., law, healthcare, finance, education) and includes evaluation scripts, basic metrics, and a reproducible deployment pipeline.

 Key Features

* Multi-language document ingestion and chunking
* Cross-lingual embedding alignment (supports HuggingFace sentence-transformers, OpenAI embeddings, etc.)
* Vector DB support (Chroma by default; easily switchable to Pinecone / Weaviate)
* Translation and cultural-context-aware pipeline
* Language detection and routing for queries
* Retrieval + context-aware generation (LLM)
* Basic evaluation (retrieval accuracy, latency, RAGAS-like scoring)
* Demo UI (Streamlit / Gradio)

 Tech Stack 

* Python 3.9+
* SentenceTransformers (HuggingFace)
* langdetect / fasttext (language detection)
* Chroma (vector DB) — or Pinecone / Weaviate
* OpenAI / local LLM or HuggingFace inference (for generation & translation)
* Transformers, sentence-transformers
* Streamlit or Gradio for demo
* PyPDF2 / pdfplumber for PDFs, Tika or OCR for images

 
 Evaluation & Metrics

* Retrieval accuracy (Precision\@k / Recall\@k)
* Latency (average query time)
* Human evaluation for cultural context and translation quality
* Optional RAGAS-like metric (if you have ground-truth QA pairs)

 Deployment

 Streamlit app: `streamlit run src/app.py`
* Gradio demo: `python src/app.py` (if Gradio)
* Hugging Face Spaces: push the repo and add `requirements.txt` + `app.py` (HF will auto-deploy)




Author: Venkata Varshith Narayanam
Email: venkatavarshithnarayanam@gmail.com

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.

