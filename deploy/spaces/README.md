# Deploy to Hugging Face Spaces (Streamlit)

1. Create a new Space → **Type**: Streamlit.
2. Upload all files from this project.
3. (Optional) In **Settings → Secrets**, add `OPENAI_API_KEY` if you plan to use OpenAI.
4. Spaces will install from `requirements.txt` and run `app.py` automatically.

Troubleshooting:
- If model downloads time out, pin smaller models (e.g., `flan-t5-small`) or switch to OpenAI.
- For persistence, note that Spaces use ephemeral storage; consider remote vector DB (e.g., Pinecone) for production.
