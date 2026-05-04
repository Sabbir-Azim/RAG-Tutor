import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


load_dotenv()


def get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


API_KEY = get_config_value("OPENAI_API_KEY")
DEFAULT_MODEL = get_config_value("OPENAI_MODEL") or "gpt-4o-mini"
MAX_OUTPUT_TOKENS = int(get_config_value("OPENAI_MAX_OUTPUT_TOKENS") or 180)
TEMPERATURE = float(get_config_value("OPENAI_TEMPERATURE") or 0.25)
VECTOR_STORE_ID = (
    get_config_value("OPENAI_VECTOR_STORE_ID")
    or get_config_value("VECTOR_STORE_ID")
)

client = OpenAI(api_key=API_KEY) if API_KEY else None

INITIAL_MESSAGE = """
Hi! I'm your RAG Tutor, your personal friendly assistant. How can I help you today?
"""

INSTRUCTIONS = """
You are RAG Tutor for a multimodal RAG course.
Answer briefly and practically. Keep most replies under 120 words unless code is requested.
Use the attached vector store when available. If the course files do not cover the answer, say so clearly.
Focus on multimodal RAG, vector stores, embeddings, retrieval design, evaluation, and deployment.
Do not invent course files, lecture names, credentials, metrics, or citations.
"""

APP_CSS = """
<style>
    :root {
        --surface: #ffffff;
        --surface-soft: #f5f7fa;
        --line: #d9dee7;
        --text: #202124;
        --muted: #687385;
        --accent: #0f766e;
        --accent-strong: #0b5f59;
        --warning: #a15c07;
        --danger: #b42318;
        --sidebar: #25282d;
    }

    .stApp {
        background: var(--surface-soft);
        color: var(--text);
    }

    .block-container,
    .block-container p,
    .block-container li,
    .block-container label,
    .block-container span {
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #eef2f6;
    }

    [data-testid="stSidebar"] .stCaptionContainer,
    [data-testid="stSidebar"] p {
        color: #c7ced8;
    }

    [data-testid="stSidebar"] button {
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] button:hover {
        border-color: rgba(255, 255, 255, 0.34);
        background: rgba(255, 255, 255, 0.13);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    .rag-shell {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 16px 40px rgba(32, 33, 36, 0.06);
    }

    .rag-header {
        align-items: center;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .rag-kicker {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0 0 0.2rem;
        text-transform: uppercase;
    }

    .rag-title {
        color: var(--text);
        font-size: 1.75rem;
        font-weight: 760;
        letter-spacing: 0;
        line-height: 1.15;
        margin: 0;
    }

    .rag-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin: 0.35rem 0 0;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        justify-content: flex-end;
    }

    .status-pill {
        align-items: center;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--muted);
        display: inline-flex;
        font-size: 0.8rem;
        font-weight: 650;
        gap: 0.4rem;
        padding: 0.34rem 0.62rem;
        white-space: nowrap;
    }

    .status-pill.ready {
        background: #eef8f6;
        border-color: #b8ded9;
        color: var(--accent-strong);
    }

    .status-pill.warn {
        background: #fff7ed;
        border-color: #fed7aa;
        color: var(--warning);
    }

    .status-dot {
        border-radius: 999px;
        display: inline-block;
        height: 0.48rem;
        width: 0.48rem;
    }

    .status-dot.ready {
        background: var(--accent);
    }

    .status-dot.warn {
        background: var(--warning);
    }

    .divider {
        border-top: 1px solid var(--line);
        margin: 1rem 0 0;
    }

    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        color: var(--text);
        box-shadow: 0 8px 22px rgba(32, 33, 36, 0.045);
        padding: 0.75rem 0.9rem;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: var(--text) !important;
    }

    [data-testid="stChatMessage"] code,
    [data-testid="stChatMessage"] pre {
        background: #f3f5f8 !important;
        color: #111827 !important;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 8px;
        border-color: #cfd6df;
    }

    .quick-prompt button {
        background: #ffffff !important;
        border: 1px solid #cfd6df !important;
        color: var(--text) !important;
        min-height: 3.1rem;
        text-align: left;
    }

    .quick-prompt button:hover {
        background: #eef8f6 !important;
        border-color: #8fc9c2 !important;
    }

    .quick-prompt button p,
    .quick-prompt button span,
    .quick-prompt button div {
        color: var(--text) !important;
    }

    .sidebar-brand {
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1rem;
        padding-bottom: 1rem;
    }

    .sidebar-brand h2 {
        color: #ffffff;
        font-size: 1.18rem;
        letter-spacing: 0;
        margin: 0;
    }

    .sidebar-brand p {
        margin: 0.3rem 0 0;
    }

    .metric-strip {
        display: grid;
        gap: 0.55rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin: 0.85rem 0 0.35rem;
    }

    .metric-box {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 0.65rem;
    }

    .metric-box span {
        color: #c7ced8;
        display: block;
        font-size: 0.72rem;
    }

    .metric-box strong {
        color: #ffffff;
        display: block;
        font-size: 0.95rem;
        margin-top: 0.15rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    @media (max-width: 760px) {
        .rag-header {
            align-items: flex-start;
            flex-direction: column;
        }

        .status-row {
            justify-content: flex-start;
        }

        .rag-title {
            font-size: 1.45rem;
        }
    }
</style>
"""

SAMPLE_PROMPTS = [
    "Explain the multimodal RAG pipeline in four practical stages.",
    "How should I chunk image-heavy course notes for retrieval?",
    "What should I evaluate in a RAG system before deployment?",
]


st.set_page_config(
    page_title="RAG Tutor",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def mask_identifier(value: str | None) -> str:
    if not value:
        return "Not set"
    if len(value) <= 12:
        return value
    return f"{value[:6]}...{value[-4:]}"


def ensure_state() -> None:
    defaults = {
        "messages": [{
            "role": "assistant",
            "content": INITIAL_MESSAGE.strip(),
        }],
        "previous_response_id": None,
        "queued_prompt": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation() -> None:
    st.session_state.messages = [{
        "role": "assistant",
        "content": INITIAL_MESSAGE.strip(),
    }]
    st.session_state.previous_response_id = None
    st.rerun()


def response_kwargs() -> dict:
    kwargs = {
        "model": DEFAULT_MODEL,
        "instructions": INSTRUCTIONS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
    }

    if VECTOR_STORE_ID:
        kwargs["tools"] = [{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
        }]

    return kwargs


def ask_bot(user_prompt: str) -> str:
    if client is None:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file or Streamlit secrets.")

    kwargs = response_kwargs()

    try:
        if st.session_state.previous_response_id:
            resp = client.responses.create(
                previous_response_id=st.session_state.previous_response_id,
                input=user_prompt,
                **kwargs,
            )
        else:
            resp = client.responses.create(input=user_prompt, **kwargs)
    except OpenAIError as exc:
        message = str(exc)
        if st.session_state.previous_response_id and "previous_response_id" in message:
            st.session_state.previous_response_id = None
            resp = client.responses.create(input=user_prompt, **kwargs)
        else:
            raise

    st.session_state.previous_response_id = resp.id
    output_text = getattr(resp, "output_text", "")

    if not output_text:
        return "I received a response, but it did not contain displayable text. Try asking again or reset the conversation."

    return output_text.strip()


def render_header() -> None:
    api_class = "ready" if API_KEY else "warn"
    api_label = "API ready" if API_KEY else "API key missing"

    st.markdown(
        f"""
        <div class="rag-shell">
            <div class="rag-header">
                <div>
                    <p class="rag-kicker">Multimodal RAG Course Assistant</p>
                    <h1 class="rag-title">RAG Tutor</h1>
                    <p class="rag-subtitle">Focused answers for course concepts, implementation details, and deployment decisions.</p>
                </div>
                <div class="status-row">
                    <span class="status-pill {api_class}"><span class="status-dot {api_class}"></span>{api_label}</span>
                </div>
            </div>
            <div class="divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>RAG Tutor</h2>
                <p>Course chat workspace</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-box">
                    <span>Messages</span>
                    <strong>{max(len(st.session_state.messages) - 1, 0)}</strong>
                </div>
                <div class="metric-box">
                    <span>Session</span>
                    <strong>Active</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Model: {DEFAULT_MODEL}")

        if st.button("Reset Conversation", use_container_width=True):
            reset_conversation()

        if not API_KEY:
            st.warning("OPENAI_API_KEY is not configured.")


def render_quick_prompts(disabled: bool) -> None:
    if len(st.session_state.messages) > 1:
        return

    st.write("")
    cols = st.columns(3)
    for index, prompt in enumerate(SAMPLE_PROMPTS):
        with cols[index]:
            st.markdown('<div class="quick-prompt">', unsafe_allow_html=True)
            if st.button(prompt, key=f"sample_{index}", use_container_width=True, disabled=disabled):
                st.session_state.queued_prompt = prompt
            st.markdown("</div>", unsafe_allow_html=True)


def render_messages() -> None:
    st.write("")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_prompt(prompt: str) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask_bot(clean_prompt)
            except (OpenAIError, ValueError) as exc:
                response = f"Sorry, I ran into an error: {exc}"
            except Exception as exc:
                response = f"Sorry, something unexpected happened: {exc}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    ensure_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)

    render_sidebar()
    render_header()
    render_quick_prompts(disabled=client is None)
    render_messages()

    queued_prompt = st.session_state.queued_prompt
    st.session_state.queued_prompt = None

    typed_prompt = st.chat_input(
        "Ask about multimodal RAG, retrieval design, evaluation, or deployment...",
        disabled=client is None,
    )

    prompt = queued_prompt or typed_prompt
    if prompt:
        handle_prompt(prompt)


if __name__ == "__main__":
    main()
