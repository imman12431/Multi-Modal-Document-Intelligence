import json
import math
import re
import string
import numpy as np
import faiss
import os
import boto3
from botocore.exceptions import ClientError


# --------------------------------------------------
# Hybrid search weight
# 0.0 = pure vector search (ignore keywords)
# 1.0 = pure BM25 keyword search (ignore semantics)
# 0.5 = equal weight (good default)
# --------------------------------------------------

DEFAULT_ALPHA = 0.5

# BM25 tuning — standard defaults, rarely need changing
BM25_K1 = 1.5   # term frequency saturation
BM25_B  = 0.75  # document length normalisation


class VectorStore:

    # --------------------------------------------------
    # Initialize store
    # --------------------------------------------------

    def __init__(self, embedding_dim=384, region="us-east-1"):

        print("Initializing Titan embedding client...")

        self.embedding_dim = embedding_dim
        self.model_id = "amazon.titan-embed-image-v1"

        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )

        self.index = faiss.IndexFlatL2(embedding_dim)
        self.items = []

        # BM25 corpus — built when build_index() is called
        self._bm25_corpus  = []   # tokenized text per item
        self._bm25_idf     = {}   # IDF per term
        self._bm25_avg_len = 0.0  # average corpus document length

        print(f"✅ FAISS vector store initialized ({embedding_dim} dims)")

    # --------------------------------------------------
    # Embed query text via Titan
    # --------------------------------------------------

    def embed_text(self, text):

        body = {
            "inputText": text,
            "embeddingConfig": {
                "outputEmbeddingLength": self.embedding_dim
            }
        }

        try:

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )

            result = json.loads(response["body"].read())
            embedding = result.get("embedding")

            if not embedding:
                raise RuntimeError("Titan returned empty embedding")

            if len(embedding) != self.embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: "
                    f"{len(embedding)} vs {self.embedding_dim}"
                )

            return np.array(embedding, dtype=np.float32)

        except ClientError as e:
            raise RuntimeError(f"Bedrock call failed: {e}")

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
    # Build FAISS index + BM25 corpus
    # --------------------------------------------------

    def build_index(self):

        print("Building FAISS index + BM25 corpus...")

        embeddings = []

        for i, item in enumerate(self.items):

            if "embedding" not in item:
                print(f"\n🚨 Missing embedding at index {i}")
                print(json.dumps(item, indent=2)[:500])
                raise RuntimeError("Embedding missing — stop build.")

            embeddings.append(item["embedding"])

        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.reset()
        self.index.add(embeddings)

        # Build BM25 corpus from the searchable text of each item
        self._bm25_corpus = [
            _tokenize(_searchable_text(item))
            for item in self.items
        ]

        self._bm25_idf     = _compute_idf(self._bm25_corpus)
        self._bm25_avg_len = (
            sum(len(doc) for doc in self._bm25_corpus) / len(self._bm25_corpus)
            if self._bm25_corpus else 1.0
        )

        print(f"✅ FAISS index: {self.index.ntotal} vectors")
        print(f"✅ BM25 corpus: {len(self._bm25_corpus)} documents, "
              f"{len(self._bm25_idf)} unique terms, "
              f"avg length {self._bm25_avg_len:.1f} tokens")

    # --------------------------------------------------
    # Hybrid search
    # --------------------------------------------------

    def search(self, query_embedding, query_text, k=5,
               alpha=DEFAULT_ALPHA, faiss_fetch_multiplier=4):
        """
        Hybrid BM25 + vector search.

        Parameters
        ----------
        query_embedding       : np.array from embed_text()
        query_text            : raw query string (used for BM25 keyword scoring)
        k                     : number of results to return
        alpha                 : keyword weight — 0.0 = pure vector, 1.0 = pure BM25
        faiss_fetch_multiplier: fetch this many extra candidates from FAISS before
                                re-ranking, so BM25 has a reasonable pool to work with
        """

        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty — call build_index() first")

        if isinstance(query_embedding, str):
            raise RuntimeError("Search received raw string — embed first!")

        # --------------------------------------------------
        # STEP 1 — Fetch a larger candidate pool from FAISS
        # --------------------------------------------------

        fetch_k = min(k * faiss_fetch_multiplier, self.index.ntotal)

        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            fetch_k
        )

        candidate_indices = [
            int(i) for i in indices.flatten()
            if i != -1 and i < len(self.items)
        ]

        if not candidate_indices:
            return []

        # --------------------------------------------------
        # STEP 2 — Vector scores (convert L2 distance → similarity)
        # FAISS IndexFlatL2 returns squared L2 distances.
        # Convert to a 0-1 similarity: sim = 1 / (1 + distance)
        # --------------------------------------------------

        raw_distances = {
            int(idx): float(dist)
            for idx, dist in zip(indices.flatten(), distances.flatten())
            if idx != -1 and idx < len(self.items)
        }

        vector_scores = {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in raw_distances.items()
        }

        # Normalise to 0-1 across candidates
        v_min = min(vector_scores.values())
        v_max = max(vector_scores.values())
        v_range = v_max - v_min or 1.0

        vector_scores = {
            idx: (score - v_min) / v_range
            for idx, score in vector_scores.items()
        }

        # --------------------------------------------------
        # STEP 3 — BM25 scores over the candidate pool
        # --------------------------------------------------

        query_tokens = _tokenize(query_text)

        bm25_scores = {}

        for idx in candidate_indices:

            doc_tokens = self._bm25_corpus[idx]
            doc_len    = len(doc_tokens)

            score = _bm25_score(
                query_tokens, doc_tokens, doc_len,
                self._bm25_avg_len, self._bm25_idf
            )

            bm25_scores[idx] = score

        # Normalise BM25 scores to 0-1
        b_max = max(bm25_scores.values()) or 1.0

        bm25_scores = {
            idx: score / b_max
            for idx, score in bm25_scores.items()
        }

        # --------------------------------------------------
        # STEP 4 — Combine and re-rank
        # --------------------------------------------------

        combined = {}

        for idx in candidate_indices:
            v = vector_scores.get(idx, 0.0)
            b = bm25_scores.get(idx, 0.0)
            combined[idx] = (1.0 - alpha) * v + alpha * b

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # --------------------------------------------------
        # STEP 5 — Return top-k items (strip embedding field)
        # --------------------------------------------------

        results = []

        for idx, score in ranked[:k]:

            item = {
                key: val
                for key, val in self.items[idx].items()
                if key != "embedding"
            }

            item["_score"]        = round(score, 4)
            item["_vector_score"] = round(vector_scores.get(idx, 0.0), 4)
            item["_bm25_score"]   = round(bm25_scores.get(idx, 0.0), 4)

            results.append(item)

        return results

    # --------------------------------------------------
    # Save / load FAISS index
    # --------------------------------------------------

    def save(self, path="faiss_index"):
        faiss.write_index(self.index, path)
        print(f"FAISS index saved → {path}")

    def load(self, path="faiss_index"):
        self.index = faiss.read_index(path)
        print("FAISS index loaded")


# --------------------------------------------------
# BM25 helpers (module-level, no external dependencies)
# --------------------------------------------------

def _tokenize(text):
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _searchable_text(item):
    """
    Extracts the text that BM25 should score against for a given item.
    - text   → raw text chunk
    - table  → summary + raw table text
    - image  → summary only
    - page   → summary only
    """
    parts = []

    text    = item.get("text", "")
    summary = item.get("summary", "")
    caption = item.get("caption", "")

    if text:
        parts.append(text)
    if summary:
        parts.append(summary)
    if caption:
        parts.append(caption)

    return " ".join(parts)


def _compute_idf(corpus):
    """
    Compute IDF (inverse document frequency) for every term in the corpus.
    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    This is the standard BM25+ IDF formula.
    """
    N  = len(corpus)
    df = {}

    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    idf = {}

    for term, freq in df.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    return idf


def _bm25_score(query_tokens, doc_tokens, doc_len, avg_len, idf):
    """
    Compute BM25 score for a single document given the query tokens.
    """
    tf_map = {}
    for token in doc_tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    score = 0.0

    for token in query_tokens:

        if token not in idf:
            continue

        tf   = tf_map.get(token, 0)
        norm = tf * (BM25_K1 + 1) / (
            tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avg_len)
        )

        score += idf[token] * norm

    return score


# --------------------------------------------------
# Test run
# --------------------------------------------------

if __name__ == "__main__":

    store = VectorStore()
    store.load_items("data/embedded_items.json")
    store.build_index()

    test_query = "What is this document about?"

    print("\nEmbedding test query via Titan...")
    q_emb = store.embed_text(test_query)

    results = store.search(q_emb, query_text=test_query, k=3, alpha=0.5)

    print("\nHybrid search results:")
    for r in results:
        print(f"  score={r['_score']}  "
              f"vector={r['_vector_score']}  "
              f"bm25={r['_bm25_score']}  "
              f"type={r.get('type')}  "
              f"page={r.get('page')}")

    print("\nVector store ready.")