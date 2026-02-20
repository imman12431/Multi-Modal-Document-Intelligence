import base64
import os
import traceback

import streamlit as st

from vector_store import VectorStore
from llm_qa import NovaMultimodalQA
import config


# -----------------------------------------------------
# Page config
# -----------------------------------------------------

st.set_page_config(
    page_title="Document QA",
    page_icon="📄",
    layout="wide"
)


# -----------------------------------------------------
# Session state
# -----------------------------------------------------

for key, default in {
    "vector_store":  None,
    "qa_system":     None,
    "loaded":        False,
    "chat_history":  [],
    "sample_query":  None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------------------------------
# Pipeline loader — runs once, silent after success
# -----------------------------------------------------

if not st.session_state.loaded:

    with st.spinner("Initializing..."):

        try:

            if not os.path.exists(config.EMBEDDED_ITEMS_PATH):
                st.error("⚠️ Embedded items file not found. Run the pipeline first.")
                st.stop()

            vector_store = VectorStore()
            vector_store.load_items(config.EMBEDDED_ITEMS_PATH)
            vector_store.build_index()
            st.session_state.vector_store = vector_store
            st.session_state.qa_system    = NovaMultimodalQA()
            st.session_state.loaded       = True

        except Exception as e:
            st.error(f"Initialization failed: {e}")
            st.stop()


# -----------------------------------------------------
# Context rendering helper
# -----------------------------------------------------

def render_context_items(items):
    """
    Renders retrieved context items in a collapsible section.
    Images and tables show their summary; text collapsed if long.
    """

    with st.expander(f"📎 Retrieved context ({len(items)} items)", expanded=False):

        for i, item in enumerate(items):

            item_type  = item.get("type", "unknown")
            page       = item.get("page")
            page_label = f"Page {page + 1}" if page is not None else ""

            st.markdown(f"**{i + 1}. {item_type.capitalize()}** {page_label}")

            if item_type in ("image", "page"):

                summary = item.get("summary", "No summary available.")
                st.caption(summary)

            elif item_type == "table":

                summary = item.get("summary", "")
                raw     = item.get("text", "")

                if summary:
                    st.caption(summary)

                if raw:
                    with st.expander("Show raw table data", expanded=False):
                        st.text(raw)

            elif item_type == "text":

                text = item.get("text", "")

                if len(text) > 300:
                    with st.expander("Show full text", expanded=False):
                        st.markdown(text)
                else:
                    st.markdown(text)

            st.divider()


# -----------------------------------------------------
# Header — title, about, document link
# -----------------------------------------------------

st.title("📄 Multimodal Document QA")

st.markdown(
    """
    This app lets you ask questions about a PDF document using a multimodal RAG pipeline.
    Text, tables, and images are all extracted from the document, summarised, and embedded
    into a searchable vector store. When you ask a question, the most relevant content is
    retrieved using a hybrid keyword + semantic search and passed to Amazon Nova to generate
    an answer.

    📂 **Source document:** [View on Google Drive](https://drive.google.com/drive/folders/1KGxnFFPKB7O6cfUqjgkV2JlHN1BCxKEk)
    """
)

st.divider()


# -----------------------------------------------------
# Sample questions
# -----------------------------------------------------

SAMPLE_QUESTIONS = [
    "What were the key economic indicators for Qatar in 2024?",
    "Summarise the main findings from the tables in the document.",
]

if not st.session_state.chat_history:

    st.markdown("**Try a sample question:**")

    cols = st.columns(len(SAMPLE_QUESTIONS))

    for col, question in zip(cols, SAMPLE_QUESTIONS):
        with col:
            if st.button(question, use_container_width=True):
                st.session_state.sample_query = question
                st.rerun()

    st.divider()


# -----------------------------------------------------
# Chat history display
# -----------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context_items"):
            render_context_items(msg["context_items"])


# -----------------------------------------------------
# Resolve query — either from sample button or chat input
# -----------------------------------------------------

query = st.chat_input("Ask a question about the document...")

if st.session_state.sample_query:
    query = st.session_state.sample_query
    st.session_state.sample_query = None

if query:

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.chat_history.append({
        "role":    "user",
        "content": query
    })

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching..."):
                query_embedding = st.session_state.vector_store.embed_text(query)
                search_results  = st.session_state.vector_store.search(
                    query_embedding,
                    query_text=query,
                    k=5
                )

            with st.spinner("Generating answer..."):
                result = st.session_state.qa_system.generate_answer_with_context(
                    query,
                    search_results
                )

            answer = result["answer"]

            st.markdown(answer)
            render_context_items(search_results)

            st.session_state.chat_history.append({
                "role":          "assistant",
                "content":       answer,
                "context_items": search_results
            })

        except Exception:
            st.error("Something went wrong. Please try again.")
            st.code(traceback.format_exc())