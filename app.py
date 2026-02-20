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

st.title("📄 Document QA")


# -----------------------------------------------------
# Session state
# -----------------------------------------------------

for key, default in {
    "vector_store": None,
    "qa_system":    None,
    "loaded":       False,
    "chat_history": []
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

            st.session_state.qa_system = NovaMultimodalQA()
            st.session_state.loaded    = True

        except Exception as e:
            st.error(f"Initialization failed: {e}")
            st.stop()


# -----------------------------------------------------
# Context rendering helpers
# -----------------------------------------------------

def _load_image_bytes(item):
    """Return base64 image string — from memory or disk."""
    if "image" in item:
        return item["image"]
    path = item.get("path", "")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def render_context_items(items):
    """
    Renders retrieved context items in a collapsible section.
    - Images  → displayed as images
    - Tables  → raw data in a collapsed expander
    - Text    → collapsed if long (> 300 chars)
    """

    with st.expander(f"📎 Retrieved context ({len(items)} items)", expanded=False):

        for i, item in enumerate(items):

            item_type = item.get("type", "unknown")
            page      = item.get("page")
            page_label = f"Page {page + 1}" if page is not None else ""

            # Strip internal score keys before display
            clean_item = {
                k: v for k, v in item.items()
                if not k.startswith("_") and k not in ("embedding", "image", "embedded_text")
            }

            st.markdown(f"**{i + 1}. {item_type.capitalize()}** {page_label}")

            if item_type in ("image", "page"):

                image_b64 = _load_image_bytes(item)

                if image_b64:
                    caption = item.get("caption") or item.get("summary", "")[:120]
                    st.image(
                        base64.b64decode(image_b64),
                        caption=caption or None,
                        use_container_width=True
                    )
                else:
                    # Fall back to showing the summary as text
                    summary = item.get("summary", "No summary available.")
                    st.caption(summary)

            elif item_type == "table":

                summary = item.get("summary", "")
                raw     = item.get("text", "")

                if summary:
                    st.caption(f"**Summary:** {summary}")

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
# Chat history display
# -----------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context_items"):
            render_context_items(msg["context_items"])


# -----------------------------------------------------
# Query input
# -----------------------------------------------------

query = st.chat_input("Ask a question about the document...")

if query:

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.chat_history.append({
        "role":    "user",
        "content": query
    })

    # Generate and show assistant response
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
                "role":         "assistant",
                "content":      answer,
                "context_items": search_results
            })

        except Exception:
            st.error("Something went wrong. Please try again.")
            st.code(traceback.format_exc())