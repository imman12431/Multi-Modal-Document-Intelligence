import streamlit as st
import os

from vector_store import VectorStore
from llm_qa import NovaMultimodalQA
from create_embeddings import generate_multimodal_embeddings
import config


# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Modal RAG",
    layout="wide"
)

st.title("📄 Multi-Modal RAG — Qatar IMF Report")


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# Load vector store + Nova QA
# --------------------------------------------------

if not st.session_state.loaded:

    index_path = "faiss_index"

    if os.path.exists(index_path):

        with st.spinner("Loading vector store + Nova QA..."):

            try:
                store = VectorStore()
                store.load_items(config.EMBEDDED_ITEMS_PATH)
                store.load(index_path)

                qa = NovaMultimodalQA()

                st.session_state.vector_store = store
                st.session_state.qa_system = qa
                st.session_state.loaded = True

            except Exception as e:
                st.error(f"Loading failed: {e}")
                st.session_state.loaded = False


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("System Status")

    if st.session_state.loaded:

        st.success("Pipeline Ready")

        total = len(st.session_state.vector_store.items)

        st.write(f"Embedded items: {total}")

        st.markdown("---")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    else:

        st.error("Pipeline not initialized")

# --------------------------------------------------
# Chat UI
# --------------------------------------------------

if st.session_state.loaded:

    st.markdown("---")

    # Display history
    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if "metadata" in msg:

                with st.expander("Retrieved Context"):
                    for meta in msg["metadata"]:
                        st.markdown(
                            f"Type: {meta['type']} | Page: {meta['page']}"
                        )

    # User input
    query = st.chat_input("Ask about the document…")

    if query:

        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Embedding → Searching → Nova reasoning..."):

                # ----------------------------------
                # Embed query
                # ----------------------------------

                query_embedding = generate_multimodal_embeddings(
                    prompt=query
                )

                # ----------------------------------
                # Search FAISS
                # ----------------------------------

                matched_items = st.session_state.vector_store.search(
                    query_embedding,
                    k=5
                )

                # ----------------------------------
                # Nova QA
                # ----------------------------------

                result = st.session_state.qa_system.generate_answer_with_context(
                    query,
                    matched_items
                )

                st.markdown(result["answer"])

                with st.expander("Retrieved Context"):
                    for meta in result["metadata"]:
                        st.markdown(
                            f"Type: {meta['type']} | Page: {meta['page']}"
                        )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": result["metadata"]
                })

else:

    st.info("Pipeline not ready — run preprocessing scripts first.")
