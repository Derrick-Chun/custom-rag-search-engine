# Custom Hybrid RAG Engine & Document Intelligence Pipeline

An end-to-end, high-performance information retrieval (IR) and document processing pipeline built from scratch to ingest canonical LLM research literature, construct an automated relational markdown wiki, and execute hybrid (dense + sparse) vector search for context-aware query synthesis. Developed for INDENG 231.

## 📊 Architectural Capabilities & Highlights

* **Engineered-from-Scratch Pipeline (`kb_pipeline.py`):** Bypassed high-level frameworks (LangChain/LlamaIndex) to construct custom document chunking layers featuring configurable sliding-window token overlaps to preserve semantic continuity across boundary points.
* **Hybrid Vector Retrieval System:** Developed a dual-vector scoring matrix combining local deterministic sparse features (TF-IDF keyword distributions) with semantic dense passage embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via synchronized cosine-similarity computation to resolve structural keyword-miss failure modes.
* **Automated Knowledge-Web Compiler:** Implemented a graph-style indexing engine using localized regular expressions to automatically parse concepts, inject cross-paper reciprocal backlinks, and build an interconnected relational wiki with rigorous provenance tracking.
* **Latent Space Evaluation:** Integrated Principal Component Analysis (PCA) dimensional reduction within the driver suite to visually map and validate multi-document embedding clusters, ensuring uniform vector distributions.

---

## 📚 Core Academic Corpus Parsed
The ingestion pipeline processes an 8-paper canonical domain layer covering structural advancements from foundation transformers to attention-window scaling limits:
1. **Vaswani et al. (2017)** — *Attention Is All You Need* (Transformer Baselines)
2. **Devlin et al. (2018)** — *BERT* (Bidirectional Contextual Encoding)
3. **Brown et al. (2020)** — *GPT-3* (Autoregressive Scaling Limits)
4. **Lewis et al. (2020)** — *Retrieval-Augmented Generation* (Parametric Memory Foundations)
5. **Wei et al. (2022)** — *Chain-of-Thought Prompting* (Multi-Step Inference Pathways)
6. **Bai et al. (2022)** — *Constitutional AI* (Self-Critique & Alignment Vectors)
7. **Liu et al. (2023)** — *Lost in the Middle* (U-Shaped Serial Position Constraints)

---

## 🛠️ Workspace Architecture

```text
├── kb_pipeline.py        # Core Engine: Tokenizer layers, TF-IDF matrices, cosine distance math, graph linking
├── kb_main.ipynb         # Driver Suite: Dense transformer sync, semantic evaluations, cluster checks, inference UI
├── requirements.txt      # Dependency pinning (numpy, scikit-learn, sentence-transformers, openai, pdfplumber)
└── .gitignore            # Security layer protecting the runtime from heavy PDF ingestion binaries and build junk