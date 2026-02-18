import streamlit as st
import os
import traceback

from vector_store import VectorStore
from llm_qa import NovaMultimodalQA
import config


# -----------------------------------------------------
# Streamlit setup
# -----------------------------------------------------

st.set_page_config(page_title="Multimodal RAG Debug")

st.title("📄 Multimodal RAG — Debug Mode")


# -----------------------------------------------------
# Session state init
# -----------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------------------------
# DEBUG PANEL — sidebar
# -----------------------------------------------------

with st.sidebar:

    st.header("🔍 Debug Panel")

    st.write("### Expected files:")

    debug_paths = {
        "PDF": config.PDF_PATH,
        "Extracted JSON": config.CHUNKS_PATH,
        "Embedded JSON": config.EMBEDDED_ITEMS_PATH,
        "FAISS index": "faiss_index",
    }

    for label, path in debug_paths.items():
        exists = os.path.exists(path)
        st.write(f"{label}:")
        st.code(path)
        st.write("✅ Exists" if exists else "❌ Missing")
        st.markdown("---")


# -----------------------------------------------------
# Pipeline loader with debug logging
# -----------------------------------------------------

if not st.session_state.loaded:

    st.info("🔄 Attempting to initialize pipeline...")

    try:

        # ---- Check embedded file exists ----
        if not os.path.exists(config.EMBEDDED_ITEMS_PATH):
            st.error("❌ Embedded items JSON not found.")
            st.stop()

        st.write("✅ Embedded items found")

        # ---- Load vector store ----
        st.write("Loading vector store...")

        vector_store = VectorStore()
        vector_store.load_items(config.EMBEDDED_ITEMS_PATH)
        vector_store.build_index()

        st.session_state.vector_store = vector_store

        st.write("✅ Vector store ready")

        # ---- Load Nova QA ----
        st.write("Initializing Nova QA...")

        qa = NovaMultimodalQA()

        st.session_state.qa_system = qa

        st.write("✅ QA system ready")

        st.session_state.loaded = True

        st.success("🎉 Pipeline initialized successfully!")

    except Exception as e:

        st.error("🚨 Pipeline initialization failed")

        st.text(str(e))

        st.code(traceback.format_exc())

        st.session_state.loaded = False


# -----------------------------------------------------
# Main app interface
# -----------------------------------------------------

if st.session_state.loaded:

    st.markdown("---")
    st.success("✅ Pipeline ready")

    # ---- Chat history ----
    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    # ---- User input ----
    query = st.chat_input("Ask a question about the document...")

    if query:

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching + generating answer..."):

                try:

                    # FAISS search
                    results = st.session_state.vector_store.search(
                        st.session_state.vector_store.items[0]["embedding"],
                        k=5
                    )

                    st.write("Debug — retrieved items:")
                    st.json(results)

                    # Nova QA
                    answer = st.session_state.qa_system.generate_answer(
                        query,
                        results
                    )

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error("❌ Runtime error")

                    st.code(traceback.format_exc())


else:

    st.error("🚨 Pipeline not initialized")

