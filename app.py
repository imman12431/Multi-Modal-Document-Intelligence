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
## Multimodal RAG Pipeline — Project Summary

This project is an end-to-end RAG system that answers natural language questions about a PDF by understanding its text, tables, and images together. The pipeline runs offline to process and index the document, and a Streamlit app serves as the live query interface.

📂 **Example document used for this Demo:** [View on Google Drive](https://drive.google.com/drive/folders/1Cl24giYMDdK9zoMY-eaCX6_PuRK9i-Nw?usp=sharing)

## Key Features

- **Multimodal extraction** — extracts text, tables, and images from the PDF including charts and image-based tables rendered as vector drawings, which standard extractors miss entirely
- **Layout-aware clustering** — separates adjacent visual elements on the same page using bounding-box gap checks, text separator vetoes, and size caps so side-by-side charts are never merged into one
- **Nova-powered summarisation** — every image and table is described by Amazon Nova Pro before embedding, creating a searchable text representation of visual content
- **Hybrid BM25 + vector search** — combines dense semantic retrieval with IDF-weighted exact keyword matching, boosting results that contain the precise terms used in the question
- **Two-stage Nova architecture** — Nova is used once during ingestion to summarise visuals for retrieval, and again at query time to reason over the original images and raw table data when generating answers
    """
)

col1, col2 = st.columns(2)

with col1:
    st.image(
        "assets/table.png",
        caption="Example of tabular data in the PDF",
        width=500
    )

with col2:
    st.image(
        "assets/figure.png",
        caption="Example of figure data in the PDF",
        width=300
    )

st.divider()


# -----------------------------------------------------
# Sample questions
# -----------------------------------------------------

SAMPLE_QUESTIONS = [
    "What was Qatar's nominal GDP in 2020 in billions of Qatari Riyals",
    "Who had the largest share of bank domestic credit in October 2024",
]

st.markdown("**Try a sample question:**")

cols = st.columns(len(SAMPLE_QUESTIONS))

for col, question in zip(cols, SAMPLE_QUESTIONS):
    with col:
        if st.button(question, use_container_width=True):
            st.session_state.sample_query = question

st.divider()


# -----------------------------------------------------
# Resolve query — either from sample button or chat input
# -----------------------------------------------------

query = st.chat_input("Ask a question about the document...")

if st.session_state.sample_query:
    query = st.session_state.sample_query
    st.session_state.sample_query = None


# -----------------------------------------------------
# Chat history display
# -----------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context_items"):
            render_context_items(msg["context_items"])


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