import json
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer


class VectorStore:

    # --------------------------------------------------
    # Initialize store
    # --------------------------------------------------

    def __init__(self, embedding_dim=384):

        print("Loading embedding model...")

        # default 384-dim model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.items = []

        print(f"✅ FAISS vector store initialized ({embedding_dim} dims)")

    # --------------------------------------------------
    # Embed query text
    # --------------------------------------------------

    def embed_text(self, text):

        embedding = self.embedding_model.encode(text)

        if len(embedding) != self.embedding_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: "
                f"{len(embedding)} vs {self.embedding_dim}"
            )

        return embedding

    # --------------------------------------------------
    # Load embedded multimodal items
    # --------------------------------------------------

    def load_items(self, embedded_items_path):

        if not os.path.exists(embedded_items_path):
            raise FileNotFoundError("Embedded items JSON not found")

        print("Loading embedded items...")

        with open(embedded_items_path, "r") as f:
            self.items = json.load(f)

        print(f"✅ Loaded {len(self.items)} items")

    # --------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------

    def build_index(self):

        print("Building FAISS index...")

        embeddings = []

        for i, item in enumerate(self.items):

            if "embedding" not in item:

                print("\n🚨 Missing embedding detected!")
                print(f"Item index: {i}")
                print(json.dumps(item, indent=2)[:1000])

                raise RuntimeError("Embedding missing — stop build.")

            embeddings.append(item["embedding"])

        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.reset()
        self.index.add(embeddings)

        print(f"✅ Index contains {self.index.ntotal} vectors")

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(self, query_embedding, k=5):

        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty")

        if isinstance(query_embedding, str):
            raise RuntimeError(
                "Search received raw string — embed first!"
            )

        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )

        matched_items = []

        for i in indices.flatten():

            item = {
                key: val
                for key, val in self.items[i].items()
                if key != "embedding"
            }

            matched_items.append(item)

        return matched_items

    # --------------------------------------------------
    # Save / load index
    # --------------------------------------------------

    def save(self, path="faiss_index"):

        faiss.write_index(self.index, path)
        print(f"FAISS index saved → {path}")

    def load(self, path="faiss_index"):

        self.index = faiss.read_index(path)
        print("FAISS index loaded")


# --------------------------------------------------
# Test run
# --------------------------------------------------

if __name__ == "__main__":

    store = VectorStore()

    store.load_items("data/embedded_items.json")
    store.build_index()

    test_query = "What is this document about?"

    print("\nEmbedding test query...")

    q_emb = store.embed_text(test_query)

    results = store.search(q_emb, k=3)

    print("\nSearch results:")
    print(json.dumps(results, indent=2))

    print("\nVector store ready.")
