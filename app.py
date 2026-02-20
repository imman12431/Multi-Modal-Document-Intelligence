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

---

### Document Ingestion

Content is extracted in four passes per page using PyMuPDF: text chunking, Tabula table extraction, embedded image extraction, and a vector drawing pass using `page.get_drawings()` that captures charts and image-based tables invisible to standard methods. A custom clustering algorithm separates adjacent visual elements using bounding-box gap checks, text separator vetoes, and size caps so nearby charts are never merged into a single crop.

---

### Summarisation

Every image and table is summarised by Amazon Nova Pro before embedding, since raw image vectors optimise for visual similarity rather than relevance to a text question. Summaries are generated in parallel with throttling retry logic, and a content-hash cache persists results to disk so unchanged items are never re-summarised.

---

### Embedding

All items are embedded as text using Amazon Titan Embed Image v1, producing 384-dimensional vectors. Text chunks embed directly, tables and images embed their Nova summary. This keeps the entire vector space text-to-text so question and document embeddings are always semantically comparable.

---

### Hybrid Search

FAISS retrieves a candidate pool using dense vector similarity, which BM25 then re-ranks using IDF-weighted exact keyword matching. Both scores are normalised and combined at equal weight, balancing broad semantic retrieval with precise keyword boosting for domain-specific terms.

---

### Answer Generation

Nova Pro receives text as context blocks, tables as summary plus raw data, and images as actual inline PNG data — not just their summary. This two-stage Nova architecture separates retrieval (text-to-text similarity) from comprehension (vision model reasoning over actual visuals).

---

### Application

The Streamlit app presents a chat interface with sample questions and a collapsible context panel per response, showing which document items drove each answer.

    📂 **Source document:** [View on Google Drive](https://drive.google.com/drive/folders/1KGxnFFPKB7O6cfUqjgkV2JlHN1BCxKEk)
    """
)

st.divider()


# -----------------------------------------------------
# Sample questions
# -----------------------------------------------------

SAMPLE_QUESTIONS = [
    "What was Qatar's nominal GDP in 2020 in billions of Qatari Riyal",
    "Who had the largest share of bank domestic credit in October 2024",
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