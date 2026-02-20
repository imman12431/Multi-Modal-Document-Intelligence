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

    debug_paths = {
        "PDF": config.PDF_PATH,
        "Extracted JSON": config.CHUNKS_PATH,
        "Embedded JSON": config.EMBEDDED_ITEMS_PATH,
    }

    for label, path in debug_paths.items():
        exists = os.path.exists(path)
        st.write(label)
        st.code(path)
        st.write("✅ Exists" if exists else "❌ Missing")
        st.markdown("---")


# -----------------------------------------------------
# Pipeline loader
# -----------------------------------------------------

if not st.session_state.loaded:

    st.info("🔄 Initializing pipeline...")

    try:

        if not os.path.exists(config.EMBEDDED_ITEMS_PATH):
            st.error("Embedded items file missing.")
            st.stop()

        st.write("Loading vector store...")

        vector_store = VectorStore()
        vector_store.load_items(config.EMBEDDED_ITEMS_PATH)
        vector_store.build_index()

        st.session_state.vector_store = vector_store

        st.success("Vector store ready")

        st.write("Initializing Nova QA...")

        qa = NovaMultimodalQA()

        st.session_state.qa_system = qa

        st.success("QA system ready")

        st.session_state.loaded = True

        st.success("🎉 Pipeline initialized!")

    except Exception as e:

        st.error("Pipeline initialization failed")

        st.text(str(e))
        st.code(traceback.format_exc())

        st.session_state.loaded = False


# -----------------------------------------------------
# Main app interface
# -----------------------------------------------------

if st.session_state.loaded:

    st.success("Pipeline ready")
    st.markdown("---")

    # show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # user input
    query = st.chat_input("Ask a question about the document...")

    if query:

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("assistant"):

            try:

                st.write("=== DEBUG STEP 1 — QUERY ===")
                st.code(query)

                # -------------------------------------------------
                # STEP 2 — Embed query
                # -------------------------------------------------

                st.write("Embedding query...")

                query_embedding = st.session_state.vector_store.embed_text(query)

                st.write("Embedding size:")
                st.code(len(query_embedding))

                # -------------------------------------------------
                # STEP 3 — Vector search
                # -------------------------------------------------

                st.write("Running vector search...")

                search_results = st.session_state.vector_store.search(
                    query_embedding,
                    query_text=query,
                    k=5
                )

                st.write("Retrieved items:")
                st.json(search_results)

                # -------------------------------------------------
                # STEP 4 — Nova QA
                # -------------------------------------------------

                st.write("Calling Nova QA...")

                try:

                    result = st.session_state.qa_system.generate_answer_with_context(
                        query,
                        search_results
                    )

                except Exception as e:

                    st.error(f"QA failure: {e}")
                    raise

                answer = result["answer"]

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception:

                st.error("❌ Runtime failure")
                st.code(traceback.format_exc())

else:

    st.error("🚨 Pipeline not initialized")