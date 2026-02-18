import json
import numpy as np
import faiss
import os


class VectorStore:

    # --------------------------------------------------
    # Initialize store
    # --------------------------------------------------

    def __init__(self, embedding_dim=384):

        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.items = []

        print(f"FAISS vector store initialized ({embedding_dim} dims)")

    # --------------------------------------------------
    # Load embedded multimodal items
    # --------------------------------------------------

    def load_items(self, embedded_items_path):

        if not os.path.exists(embedded_items_path):
            raise FileNotFoundError("Embedded items JSON not found")

        print("Loading embedded items...")

        with open(embedded_items_path, "r") as f:
            self.items = json.load(f)

        print(f"Loaded {len(self.items)} items")

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
                print("Item contents:")
                print(json.dumps(item, indent=2)[:1000])  # truncate long output

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

        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )

        matched_items = [
            {k: v for k, v in self.items[i].items() if k != "embedding"}
            for i in indices.flatten()
        ]

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

    # Example only — assumes embeddings JSON exists
    store.load_items("data/embedded_items.json")
    store.build_index()

    print("\nVector store ready.")
