# EduRAG Learning Path (Assignment-15)

A complete **RAG system** that creates **personalized learning paths** by retrieving and organizing educational content based on **student level**, **learning style**, and **progress tracking**. It satisfies the requirements in the assignment brief (educational content processing, personalized path generation, progress tracking, learning style accommodation, competency-based sequencing, adaptive difficulty, knowledge-gap remediation, evaluation, and a working demo via Streamlit).

## 🚀 Features (Mapped to Requirements)
- **Educational content processing & categorization**: Ingest PDFs/Markdown/Text/URLs, chunk & tag resources by topic/type/difficulty.
- **Retrieval with Vector DB**: Uses **Chroma** + **Sentence-Transformers** (default) or **OpenAI embeddings** (optional).
- **Context-aware generation**: Answer questions with retrieved context using **FLAN-T5** (local) or **OpenAI GPT** (optional).
- **Personalized learning paths**: VARK-style **learning-style** detection + **competency graph** → sequenced plan per student level.
- **Student progress tracking**: SQLite DB stores progress, quiz results, mastery estimates, and adapts next steps.
- **Difficulty adaptation**: EWMA-based mastery & item difficulty to up/down-shift recommended content.
- **Knowledge gap remediation**: Detect weak competencies; inject targeted remedial resources.
- **Evaluation**: Simple retrieval metrics (`precision@k`, `recall@k`, latency); optional RAGAS scaffold.
- **UX**: Streamlit app with tabs: Ingest, Student Profile, Learning Style Quiz, Plan, Learn/Chat, Progress, Evaluate.

## 🧩 Project Structure
```
EduRAG-LearningPath/
├─ app.py                      # Streamlit UI (demo)
├─ requirements.txt
├─ .env.example                # Put OPENAI_API_KEY here if you want OpenAI
├─ config.yaml                 # Competency graph + domain settings
├─ rag/
│  ├─ ingest.py                # Load, chunk, tag, and persist to Chroma
│  ├─ embeddings.py            # OpenAI or HF Sentence-Transformers
│  ├─ vectorstore.py           # Chroma wrapper
│  ├─ retriever.py             # Top-k semantic retrieval
│  ├─ generator.py             # OpenAI or FLAN-T5 small text generation
├─ personalization/
│  ├─ learning_styles.py       # VARK quiz & scoring
│  ├─ path_planner.py          # Competency-based sequencing & adaptation
│  ├─ progress.py              # SQLite models and utilities
│  ├─ quiz.py                  # Simple quiz creation & grading
├─ evaluation/
│  ├─ eval_rag.py              # Basic eval (P@k, R@k, latency) + scaffold for RAGAS
│  └─ sample_eval.csv          # Sample qrels-like annotations
├─ data/
│  └─ sample/
│     ├─ content/python_intro.txt
│     ├─ content/python_control_flow.txt
│     ├─ content/python_functions.txt
│     └─ links.json           # Example YouTube/article links by topic
└─ deploy/
   └─ spaces/README.md         # Hugging Face Spaces deploy steps
```

## 🛠️ Setup (Local)
1. **Python** 3.10+ recommended. Create a venv and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. (Optional) **OpenAI**: copy `.env.example` → `.env` and set `OPENAI_API_KEY=...`.
3. **Run the demo**:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL (shown in terminal). Use the **Ingest** tab to index sample content or upload your own.

## ☁️ 1-Click Deploy (Hugging Face Spaces)
- Create a new Space (**Streamlit**).
- Upload the repository files.
- In **Settings → Secrets**, add `OPENAI_API_KEY` if using OpenAI.
- Spaces will auto-install from `requirements.txt` and launch `app.py`.

## 🔧 Configuration
- Edit `config.yaml` to define competencies, prerequisites, and default difficulty per competency.
- Vector store persistence is under `.chroma/` (created at runtime).

## 🧪 Evaluation
- Use **Evaluate** tab to compute `precision@k` and `recall@k` on `evaluation/sample_eval.csv`.
- Add your own annotations (query, relevant_doc_ids list) to grow the eval set.
- Optional RAGAS scaffold included in code comments.

## 📦 Notes
- Default models are light-weight and can run on CPU.
- FLAN-T5 small is used for local generation (downloads at first run). For higher quality, switch to OpenAI in the sidebar.
- This demo is domain-configured for **Education → Python Basics** but you can replace the sample content with your own domain data.


## 🔑 Using Gemini (Google Generative AI)
- Install deps (already in `requirements.txt`): `google-generativeai`
- Copy `.env.example` → `.env` and set `GEMINI_API_KEY=...`
- In the Streamlit sidebar, choose:
  - **Embedding provider** → `gemini` (model default: `text-embedding-004`)
  - **Generator** → `gemini` (model default: `gemini-1.5-flash`)
