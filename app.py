import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from rag_pipeline import ask_claude, retrieve_chunks

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Apollo Hospitals Chennai — Clinical Policy Assistant",
    page_icon="🏥",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling — clean, clinical white/blue palette
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main container */
        .block-container { max-width: 780px; padding-top: 2rem; }

        /* Header bar */
        .header-bar {
            background-color: #003087;
            color: white;
            padding: 1.1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }
        .header-bar h1 { font-size: 1.25rem; margin: 0; font-weight: 600; }
        .header-bar p  { font-size: 0.85rem; margin: 0.25rem 0 0; opacity: 0.85; }

        /* Chat bubbles */
        .user-bubble {
            background-color: #e8f0fe;
            border-left: 4px solid #003087;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 0.75rem 0 0.25rem;
            font-size: 0.95rem;
        }
        .assistant-bubble {
            background-color: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 0.25rem 0 0.5rem;
            font-size: 0.95rem;
        }

        /* Source pills */
        .source-pill {
            display: inline-block;
            background-color: #eaf4fb;
            color: #003087;
            border: 1px solid #b8d4ea;
            border-radius: 12px;
            padding: 0.2rem 0.75rem;
            font-size: 0.78rem;
            margin: 0.2rem 0.25rem 0.2rem 0;
        }
        .source-label {
            font-size: 0.78rem;
            color: #6c757d;
            margin: 0.5rem 0 0.2rem;
        }

        /* Input area */
        .stTextInput > div > div > input {
            border-radius: 6px;
            border: 1px solid #ced4da;
            font-size: 0.95rem;
        }

        /* Submit button */
        div[data-testid="stForm"] button[type="submit"] {
            background-color: #003087;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.45rem 1.5rem;
            font-size: 0.9rem;
        }

        /* Disclaimer footer */
        .disclaimer {
            font-size: 0.75rem;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            padding-top: 0.75rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-bar">
        <h1>🏥 Apollo Hospitals Chennai — Clinical Policy Assistant</h1>
        <p>Ask questions about hospital policies, discharge procedures, billing, and infection control protocols.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state — preserve conversation history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------
for turn in st.session_state.messages:
    st.markdown(
        f'<div class="user-bubble">🙋 <strong>You:</strong> {turn["question"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="assistant-bubble">💬 <strong>Assistant:</strong><br>{turn["answer"]}</div>',
        unsafe_allow_html=True,
    )
    if turn["sources"]:
        st.markdown('<p class="source-label">Sources</p>', unsafe_allow_html=True)
        pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in turn["sources"])
        st.markdown(pills, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form — clears after submission
# ---------------------------------------------------------------------------
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input(
        label="Your question",
        placeholder="e.g. What documents are required for patient admission?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Retrieving relevant policies and generating answer…"):
        chunks = retrieve_chunks(question.strip())
        answer = ask_claude(question.strip(), chunks)
        sources = list(dict.fromkeys(c["source"] for c in chunks))

    st.session_state.messages.append(
        {"question": question.strip(), "answer": answer, "sources": sources}
    )
    st.rerun()

# ---------------------------------------------------------------------------
# Suggested questions (shown only before first interaction)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("#### Suggested questions")
    suggestions = [
        "What is the ICU discharge procedure?",
        "What documents are required for patient admission?",
        "What are the visiting hours for ICU patients?",
        "How is patient data confidentiality maintained?",
        "What payment methods are accepted at discharge?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"suggestion_{i}"):
            with st.spinner("Retrieving relevant policies and generating answer…"):
                chunks = retrieve_chunks(suggestion)
                answer = ask_claude(suggestion, chunks)
                sources = list(dict.fromkeys(c["source"] for c in chunks))
            st.session_state.messages.append(
                {"question": suggestion, "answer": answer, "sources": sources}
            )
            st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer">
        ⚠️ This assistant provides information based on Apollo Hospitals Chennai's internal policy documents.
        It is intended to support — not replace — clinical judgement.
        Always verify critical decisions with the relevant department head or official documentation.
    </div>
    """,
    unsafe_allow_html=True,
)
