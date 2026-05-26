import os
import re
import json
from collections import Counter
import numpy as np
import pdfplumber

def load_pdf_text(pdf_path):
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            parts.append(t)
    return "\n".join(parts)

def load_corpus(raw_dir):
    corpus = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(raw_dir, fname)
        doc_id = os.path.splitext(fname)[0]
        corpus[doc_id] = load_pdf_text(path)
    return corpus

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def chunk_document(text, chunk_size=300, overlap=60):
    tokens = tokenize(text)
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
    return chunks

class TfIdfVectorizer:
    def __init__(self, min_count=2):
        self.min_count = int(min_count)
        self.vocab = None    
        self.word2idx = None    
        self.idf = None         

    def fit(self, chunks):
        global_counter = Counter()
        per_chunk_tokens = []
        for c in chunks:
            toks = tokenize(c)
            per_chunk_tokens.append(toks)
            global_counter.update(toks)
        vocab = sorted([w for w, c in global_counter.items() if c >= self.min_count])
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        N = len(chunks)
        df = np.zeros(len(vocab), dtype=np.float64)
        for toks in per_chunk_tokens:
            seen = set(toks)
            for w in seen:
                if w in self.word2idx:
                    df[self.word2idx[w]] += 1
        self.idf = np.log((N + 1) / (df + 1)) + 1.0
        return self

    def transform(self, chunks):
        V = len(self.vocab)
        X = np.zeros((len(chunks), V), dtype=np.float64)
        for i, c in enumerate(chunks):
            toks = tokenize(c)
            cnt = Counter(toks)
            for w, n in cnt.items():
                j = self.word2idx.get(w)
                if j is not None:
                    X[i, j] = n * self.idf[j]
            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] /= norm
        return X
    def fit_transform(self, chunks):
        return self.fit(chunks).transform(chunks)

def cosine_topk(query_vec, doc_matrix, k=5):
    sims = doc_matrix @ query_vec
    order = np.argsort(sims)[::-1][:k]
    return order, sims[order]

class KnowledgeBase:
    def __init__(self, raw_dir, wiki_dir):
        self.raw_dir = raw_dir
        self.wiki_dir = wiki_dir
        os.makedirs(wiki_dir, exist_ok=True)
        self.corpus = None     
        self.chunks = None     
        self.chunk_meta = None      
        self.vectorizer = None
        self.doc_matrix = None     
        self.dense_model = None    
        self.dense_matrix = None   

    def ingest(self, chunk_size=300, overlap=60):
        self.corpus = load_corpus(self.raw_dir)
        self.chunks = []
        self.chunk_meta = []
        for doc_id, text in self.corpus.items():
            doc_chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)
            for ci, chunk in enumerate(doc_chunks):
                self.chunks.append(chunk)
                self.chunk_meta.append({"doc_id": doc_id, "chunk_idx": ci})
        self.vectorizer = TfIdfVectorizer(min_count=2)
        self.doc_matrix = self.vectorizer.fit_transform(self.chunks)
        return self

    def fit_dense(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.dense_model = SentenceTransformer(model_name)
        embs = self.dense_model.encode(
            self.chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self.dense_matrix = embs / norms
        return self

    def retrieve(self, query, k=5, mode='tfidf', alpha=0.5):
        if mode == 'tfidf':
            q_vec = self.vectorizer.transform([query])[0]
            idx, sim = cosine_topk(q_vec, self.doc_matrix, k=k)
        elif mode == 'dense':
            if self.dense_matrix is None:
                raise RuntimeError(
                    "Dense backend not fitted. Call kb.fit_dense() first."
                )
            q_emb = self.dense_model.encode([query], convert_to_numpy=True)[0]
            n = np.linalg.norm(q_emb)
            if n > 0:
                q_emb = q_emb / n
            idx, sim = cosine_topk(q_emb, self.dense_matrix, k=k)

        elif mode == 'hybrid':
            if self.dense_matrix is None:
                raise RuntimeError(
                    "Dense backend not fitted. Call kb.fit_dense() first."
                )
            q_tfidf = self.vectorizer.transform([query])[0]
            tfidf_scores = self.doc_matrix @ q_tfidf

            q_emb = self.dense_model.encode([query], convert_to_numpy=True)[0]
            n = np.linalg.norm(q_emb)
            if n > 0:
                q_emb = q_emb / n
            dense_scores = self.dense_matrix @ q_emb

            def _minmax(x):
                lo, hi = x.min(), x.max()
                return (x - lo) / (hi - lo + 1e-12)
            combined = alpha * _minmax(dense_scores) + (1 - alpha) * _minmax(tfidf_scores)
            order = np.argsort(combined)[::-1][:k]
            idx, sim = order, combined[order]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        hits = []
        for i, s in zip(idx, sim):
            meta = self.chunk_meta[i]
            hits.append({
                "doc_id": meta["doc_id"],
                "chunk_idx": meta["chunk_idx"],
                "score": float(s),
                "text": self.chunks[i],
            })
        return hits

    def compile_wiki(self, llm_callable=None):
        if llm_callable is None:
            llm_callable = _extractive_summary
        written = []
        index_lines = ["# Knowledge Base Index", ""]
        index_lines.append(f"This wiki contains {len(self.corpus)} source articles compiled from the `raw/` directory.\n")
        index_lines.append("## Articles\n")

        for doc_id, text in self.corpus.items():
            doc_chunks = [c for c, m in zip(self.chunks, self.chunk_meta) if m["doc_id"] == doc_id]
            prompt = (
                f"You are summarising an academic paper. Source id: {doc_id}.\n"
                "Produce a markdown article with the following sections:\n"
                "1. **Title and authors** (one line)\n"
                "2. **Problem statement** (2-3 sentences)\n"
                "3. **Method** (3-5 sentences)\n"
                "4. **Key results** (3-5 bullet points)\n"
                "5. **Why it matters for RAG / LLMs** (2-3 sentences)\n\n"
                "Source text (first chunks):\n\n"
                + "\n\n".join(doc_chunks[:6])
            )
            article = llm_callable(prompt)
            article = _add_backlinks(article, list(self.corpus.keys()), current=doc_id)
            out_path = os.path.join(self.wiki_dir, f"{doc_id}.md")
            with open(out_path, "w") as f:
                f.write(article)
            written.append(out_path)
            index_lines.append(f"- [{doc_id}]({doc_id}.md)")
        idx_path = os.path.join(self.wiki_dir, "index.md")
        with open(idx_path, "w") as f:
            f.write("\n".join(index_lines))
        written.append(idx_path)
        return written

    def answer(self, question, k=5, llm_callable=None, mode='tfidf', alpha=0.5):
        hits = self.retrieve(question, k=k, mode=mode, alpha=alpha)
        context = "\n\n".join(
            f"[{h['doc_id']} chunk {h['chunk_idx']}]\n{h['text']}"
            for h in hits
        )
        prompt = (
            "You are an expert teaching assistant for an LLM systems course. "
            "Use the supplied context (which comes from research papers) to answer the question. "
            "Cite each claim with the bracketed source id like [01_attention chunk 3]. "
            "If the context is insufficient, say so.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        if llm_callable is None:
            llm_callable = _extractive_qa(question)
        answer_text = llm_callable(prompt)
        return answer_text, hits

def _extractive_summary(prompt):
    marker = "first chunks):"
    if marker in prompt:
        body = prompt.split(marker, 1)[1]
    else:
        body = prompt
    sentences = re.split(r"(?<=[.!?])\s+", body)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if not sentences:
        return "# (empty)\n"
    counter = Counter()
    for s in sentences:
        counter.update(tokenize(s))
    scored = []
    for s in sentences:
        toks = tokenize(s)
        if not toks:
            continue
        score = sum(counter[t] for t in toks) / max(1, len(toks))
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:8]]

    out = []
    out.append(f"# Article summary")
    out.append("")
    out.append("## Title and authors")
    out.append(f"_(extracted from source — see raw/ folder for full citation)_")
    out.append("")
    out.append("## Problem statement")
    out.append(top[0] if len(top) > 0 else "_(not found)_")
    out.append("")
    out.append("## Method")
    out.append(" ".join(top[1:3]) if len(top) > 2 else "_(not found)_")
    out.append("")
    out.append("## Key results")
    for s in top[3:6]:
        out.append(f"- {s}")
    out.append("")
    out.append("## Why it matters for RAG / LLMs")
    out.append(" ".join(top[6:8]) if len(top) > 6 else "_(not found)_")
    return "\n".join(out)

def _extractive_qa(question):
    def _f(prompt):
        if "Context:" in prompt:
            ctx = prompt.split("Context:", 1)[1].split("Answer:", 1)[0].strip()
        else:
            ctx = prompt
        q_tokens = set(tokenize(question))
        sentences = re.split(r"(?<=[.!?])\s+", ctx)
        scored = []
        for s in sentences:
            stoks = set(tokenize(s))
            overlap = len(q_tokens & stoks)
            if overlap > 0 and len(s.strip()) > 30:
                scored.append((overlap, s.strip()))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return "I could not find sufficient information in the knowledge base to answer this question."
        cite_pattern = re.compile(r"\[([^\]]+)\]")
        answer_lines = []
        for _, s in scored[:3]:
            pos = ctx.find(s)
            cite = "unknown"
            if pos != -1:
                window = ctx[:pos]
                m = list(cite_pattern.finditer(window))
                if m:
                    cite = m[-1].group(1)
            answer_lines.append(f"- {s}  _[{cite}]_")
        return (
            f"Based on the retrieved context, the relevant findings are:\n\n"
            + "\n".join(answer_lines)
        )
    return _f

def _add_backlinks(text, all_doc_ids, current):
    for did in all_doc_ids:
        if did == current:
            continue
        pattern = re.compile(rf"(?<!\[)\b{re.escape(did)}\b(?!\])")
        text = pattern.sub(f"[{did}]({did}.md)", text)
    return text

def make_openai_callable(model="gpt-4o-mini"):
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    def _call(prompt):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    return _call
