# RAG Tutor

RAG Tutor is a Streamlit chat app for learning multimodal retrieval-augmented generation. It uses the OpenAI Responses API and can optionally connect to an OpenAI vector store for course-grounded answers.

## Features

- Professional Streamlit chat interface
- Low-token default configuration
- Conversation continuity with `previous_response_id`
- Optional OpenAI vector store file search
- Safe handling for missing API keys and API errors

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_OUTPUT_TOKENS=180
```

Optional, if you want course-file retrieval:

```env
OPENAI_VECTOR_STORE_ID=your_vector_store_id_here
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Project Files

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - excludes secrets, virtual environments, caches, and logs

## Notes

Do not commit `.env` or any API keys to GitHub. The included `.gitignore` is configured to keep local secrets and generated files out of version control.
