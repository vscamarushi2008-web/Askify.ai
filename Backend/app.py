import os
import io
import threading
from flask import Flask, render_template, request, redirect, url_for
import pdfplumber
import docx
import pandas as pd

# NLP models
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# vector search fallbacks
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    from sklearn.neighbors import NearestNeighbors
    _HAS_FAISS = False

# ---------- Config ----------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chunking params
CHUNK_SIZE = 1500        # characters per chunk
CHUNK_OVERLAP = 300      # overlap characters between chunks

# Retrieval params
TOP_K = 5
MIN_QA_SCORE = 0.15      # filter out very low confidence answers

# Models (names)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"

app = Flask(__name__)

# ---------- In-memory stores ----------
chunks = []   # list of dicts: {"text":..., "source": filename, "page": int or None, "id": int}
embeddings = None  # numpy array of shape (n_chunks, dim)
metadata = []  # parallel list of chunk metadata (source, page)
index = None   # faiss or sklearn index
embed_dim = None

status_message = ""

# ---------- Model loading ----------
qa_pipeline = None
embed_model = None

def init_models():
    global qa_pipeline, embed_model, embed_dim, index, embeddings, metadata, chunks

    try:
        print("Loading embedding model:", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        embed_dim = embed_model.get_sentence_embedding_dimension()
        print("Embedding dim:", embed_dim)
    except Exception as e:
        print("Failed to load embedding model:", e)
        embed_model = None

    try:
        print("Loading QA model:", QA_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        print("QA pipeline ready")
    except Exception as e:
        print("Failed to load QA model:", e)
        qa_pipeline = None

    # initialize empty index structures
    embeddings = np.zeros((0, embed_dim)) if embed_dim else None
    metadata = []
    chunks = []
    if embed_dim and _HAS_FAISS:
        # create an L2 index (use inner product if you wish) — we'll use normalized dot product (cosine)
        index = faiss.IndexFlatIP(embed_dim)  # cosine search after normalization
    else:
        index = None

# initialize in a background thread to avoid blocking app start
threading.Thread(target=init_models, daemon=True).start()

# ---------- File extraction helpers (same as yours, slightly reused) ----------

def extract_text_from_pdf(path):
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                pages.append(txt)
    except Exception as e:
        print("pdfplumber error:", e)
    return pages

def extract_text_from_docx(path):
    texts = []
    try:
        doc = docx.Document(path)
        for p in doc.paragraphs:
            texts.append(p.text)
    except Exception as e:
        print("python-docx error:", e)
    return "\n".join(texts)

def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print("txt read error:", e)
        return ""

def extract_text_from_csv(path):
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        lines = []
        for _, row in df.iterrows():
            lines.append(" ".join(row.values.astype(str)))
        return "\n".join(lines)
    except Exception as e:
        print("csv read error:", e)
        return ""

# ---------- Chunking ----------
def chunk_text_charwise(text, filename, page_no=None, chunk_size_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP):
    if not text:
        return []
    text = text.replace("\r", " ").strip()
    chunks_local = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size_chars
        if end >= length:
            piece = text[start:length].strip()
            chunks_local.append({"text": piece, "source": filename, "page": page_no})
            break
        else:
            piece = text[start:end].strip()
            chunks_local.append({"text": piece, "source": filename, "page": page_no})
            start = end - overlap_chars
            if start < 0:
                start = 0
    return chunks_local

# ---------- Vector index helpers ----------
def add_embeddings_to_index(new_texts, new_meta):
    """
    new_texts: list of strings
    new_meta: list of metadata dicts aligned with new_texts
    """
    global embeddings, metadata, index, embed_model, embed_dim

    if embed_model is None:
        print("No embedding model loaded; cannot add to index.")
        return

    # compute embeddings in batches via sentence-transformers
    new_emb = embed_model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize to unit-length for cosine similarity
    norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    new_emb = new_emb / norms

    if embeddings is None or embeddings.size == 0:
        embeddings = new_emb
    else:
        embeddings = np.vstack([embeddings, new_emb])

    metadata.extend(new_meta)

    # add to index
    if _HAS_FAISS and index is not None:
        index.add(new_emb.astype(np.float32))
    else:
        # sklearn fallback: rebuild a NearestNeighbors index when needed (lazy)
        pass

def query_index(question, top_k=TOP_K):
    """
    Returns list of (meta, text, score) for top_k nearest chunks.
    Score is cosine similarity in [0,1]
    """
    global embeddings, metadata, index, embed_model

    if embed_model is None:
        raise RuntimeError("Embedding model not loaded")

    q_emb = embed_model.encode([question], convert_to_numpy=True, show_progress_bar=False)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    q = q_emb.astype(np.float32)

    if _HAS_FAISS and index is not None:
        # faiss inner product search (since we normalized vectors, IP == cosine)
        D, I = index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            results.append((metadata[idx], embeddings[idx], score))
        return results
    else:
        # sklearn fallback: brute-force cosine similarity
        if embeddings is None or embeddings.size == 0:
            return []
        # compute cosine similarities
        sims = (embeddings @ q.T).squeeze()  # (n_chunks,)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            results.append((metadata[idx], embeddings[idx], float(sims[idx])))
        return results

# ---------- Flask routes ----------
@app.route("/")
def index():
    global status_message
    msg = status_message
    status_message = ""
    return render_template("index.html", answer="", status=msg)

@app.route("/upload", methods=["POST"])
def upload():
    global chunks, status_message

    if "file" not in request.files:
        status_message = "No file in request."
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        status_message = "No file selected."
        return redirect(url_for("index"))

    filename = f.filename
    lower = filename.lower()
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    added = 0
    new_texts = []
    new_meta = []
    try:
        if lower.endswith(".pdf"):
            pages = extract_text_from_pdf(save_path)
            for i, p in enumerate(pages):
                page_chunks = chunk_text_charwise(p, filename, page_no=i+1)
                for pc in page_chunks:
                    # assign an id and append to global chunks
                    chunks.append(pc)
                    new_texts.append(pc["text"])
                    new_meta.append({"source": pc["source"], "page": pc["page"], "text": pc["text"]})
                added += len(page_chunks)
        elif lower.endswith(".docx"):
            text = extract_text_from_docx(save_path)
            doc_chunks = chunk_text_charwise(text, filename, page_no=None)
            for pc in doc_chunks:
                chunks.append(pc)
                new_texts.append(pc["text"])
                new_meta.append({"source": pc["source"], "page": pc["page"], "text": pc["text"]})
            added += len(doc_chunks)
        elif lower.endswith(".txt"):
            text = extract_text_from_txt(save_path)
            t_chunks = chunk_text_charwise(text, filename, page_no=None)
            for pc in t_chunks:
                chunks.append(pc)
                new_texts.append(pc["text"])
                new_meta.append({"source": pc["source"], "page": pc["page"], "text": pc["text"]})
            added += len(t_chunks)
        elif lower.endswith(".csv"):
            text = extract_text_from_csv(save_path)
            c_chunks = chunk_text_charwise(text, filename, page_no=None)
            for pc in c_chunks:
                chunks.append(pc)
                new_texts.append(pc["text"])
                new_meta.append({"source": pc["source"], "page": pc["page"], "text": pc["text"]})
            added += len(c_chunks)
        else:
            status_message = "Unsupported file type. Supported: pdf, docx, txt, csv."
            return redirect(url_for("index"))
    except Exception as e:
        print("Extraction error:", e)
        status_message = f"Error extracting file: {e}"
        return redirect(url_for("index"))

    # compute embeddings and add to index
    if new_texts:
        try:
            add_embeddings_to_index(new_texts, new_meta)
        except Exception as e:
            print("Embedding/index add error:", e)
            status_message = f"Uploaded {filename} but failed to index embeddings: {e}"
            return redirect(url_for("index"))

    status_message = f"Uploaded {filename}. Added {added} chunks (total chunks: {len(chunks)})."
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    global qa_pipeline, chunks

    question = request.form.get("question", "").strip()
    if not question:
        return render_template("index.html", answer="⚠ Please enter a question.", status="")

    if not chunks:
        return render_template("index.html", answer="⚠ No documents indexed. Upload files first.", status="")

    if qa_pipeline is None:
        return render_template("index.html", answer="⚠ QA model not loaded. Check server logs.", status="")

    # retrieve top-K relevant chunks
    try:
        retrieved = query_index(question, top_k=TOP_K)
    except Exception as e:
        print("Retrieval error:", e)
        return render_template("index.html", answer=f"Retrieval error: {e}", status="")

    if not retrieved:
        return render_template("index.html", answer="No relevant chunks found.", status="")

    best = {"score": -1.0, "answer": "", "source": None, "page": None}
    # Run QA on each retrieved chunk independently and pick the best-scoring answer
    for meta, vec, score in retrieved:
        ctx = meta.get("text", "")
        if not ctx or ctx.strip() == "":
            continue
        try:
            out = qa_pipeline(question=question, context=ctx)
        except Exception as e:
            print("pipeline error on chunk:", e)
            continue

        ans_text = out.get("answer", "").strip()
        ans_score = float(out.get("score", 0.0))
        if not ans_text:
            continue

        # simple safety filter
        if ans_text.lower() in ["[cls]", "[pad]"]:
            continue

        # prefer higher QA confidence; break ties by retrieval score if needed
        if ans_score > best["score"]:
            best = {
                "score": ans_score,
                "answer": ans_text,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "retrieval_score": score
            }

    if best["score"] < MIN_QA_SCORE or best["answer"] == "":
        display = "No confident answer found in the uploaded documents."
    else:
        display = f"Answer: {best['answer']}\n\nSource: {best['source']}"
        if best["page"] is not None:
            display += f" (page {best['page']})"
        display += f"\nQA confidence: {best['score']:.3f}"
        display += f"\nRetrieval similarity: {best.get('retrieval_score', 0.0):.3f}"

    return render_template("index.html", answer=display, status="")

@app.route("/clear_answer", methods=["POST"])
def clear_answer():
    return redirect(url_for("index"))

if np.__array_namespace_info__ == "_main_":
    # debug=True for local development only
    app.run(host="0.0.0.0", port=5001, debug=True)