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

st.title("📄 Multimodal RAG — Deep Debug Mode")


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
# DEBUG SIDEBAR — FILE CHECKS
# -----------------------------------------------------

with st.sidebar:

    st.header("🔍 Debug Panel")

    debug_paths = {
        "PDF": config.PDF_PATH,
        "Extracted JSON": config.CHUNKS_PATH,
        "Embedded JSON": config.EMBEDDED_ITEMS_PATH,
        "FAISS index (if saved)": "faiss_index",
    }

    for label, path in debug_paths.items():

        exists = os.path.exists(path)

        st.write(f"### {label}")
        st.code(path)
        st.write("✅ Exists" if exists else "❌ Missing")
        st.markdown("---")


# -----------------------------------------------------
# PIPELINE LOADER
# -----------------------------------------------------

if not st.session_state.loaded:

    st.info("🔄 Initializing pipeline...")

    try:

        # ---------- Embedded file check ----------

        if not os.path.exists(config.EMBEDDED_ITEMS_PATH):
            st.error("❌ Embedded items file missing")
            st.stop()

        st.success("Embedded items found")

        # ---------- Vector store ----------

        st.write("Loading vector store...")

        vector_store = VectorStore()

        st.write("→ Loading embedded items JSON")
        vector_store.load_items(config.EMBEDDED_ITEMS_PATH)

        st.write(f"Loaded items: {len(vector_store.items)}")

        st.write("→ Building FAISS index")
        vector_store.build_index()

        st.success("Vector store ready")

        st.session_state.vector_store = vector_store

        # ---------- Nova QA ----------

        st.write("Initializing Nova QA...")

        qa = NovaMultimodalQA()

        st.session_state.qa_system = qa

        st.success("Nova QA ready")

        st.session_state.loaded = True

        st.success("🎉 Pipeline initialized successfully!")

    except Exception as e:

        st.error("🚨 Pipeline initialization failed")

        st.text(str(e))
        st.code(traceback.format_exc())

        st.session_state.loaded = False


# -----------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------

if st.session_state.loaded:

    st.markdown("---")
    st.success("✅ Pipeline ready")

    # ---------- Chat history ----------

    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    # ---------- User input ----------

    query = st.chat_input("Ask a question about the document...")

    if query:

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching + generating answer..."):

                try:

                    # =================================================
                    # DEBUG STEP 1 — USER QUERY
                    # =================================================

                    st.write("=== DEBUG STEP 1 — QUERY ===")
                    st.code(query)

                    # =================================================
                    # DEBUG STEP 2 — VECTOR SEARCH
                    # =================================================

                    st.write("=== DEBUG STEP 2 — VECTOR SEARCH ===")

                    search_results = st.session_state.vector_store.search(query, k=5)

                    st.write("Results type:")
                    st.code(type(search_results))

                    st.write("Result count:")
                    st.code(len(search_results))

                    st.write("Raw search results:")
                    st.json(search_results)

                    if not search_results:
                        st.error("❌ No search results returned")
                        st.stop()

                    # =================================================
                    # DEBUG STEP 3 — FORMAT FOR NOVA
                    # =================================================

                    st.write("=== DEBUG STEP 3 — MATCHED ITEMS ===")

                    matched_items = []

                    for i, r in enumerate(search_results):

                        st.write(f"Processing result #{i}")

                        item = r.get("chunk", r)

                        st.json(item)

                        matched_items.append(item)

                    st.write("Final matched items:")
                    st.json(matched_items)

                    # =================================================
                    # DEBUG STEP 4 — NOVA CALL
                    # =================================================

                    st.write("=== DEBUG STEP 4 — NOVA INVOCATION ===")

                    answer = st.session_state.qa_system.generate_answer(
                        query,
                        matched_items
                    )

                    st.write("Nova response:")
                    st.code(answer)

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error("❌ Runtime failure")

                    st.text(str(e))
                    st.code(traceback.format_exc())


else:

    st.error("🚨 Pipeline not initialized")

