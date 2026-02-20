import os
import json
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from summarizer import Summarizer


# ---------------------------------------------------
# Bedrock client (shared)
# ---------------------------------------------------

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "amazon.titan-embed-image-v1"
EMBED_DIM = 384


# ---------------------------------------------------
# Titan embedding call — text only
# ---------------------------------------------------
# We always embed TEXT now (either the raw chunk or the
# Nova-generated summary). This keeps FAISS doing
# text-to-text similarity at query time.
# ---------------------------------------------------

def generate_text_embedding(text):

    if not text or not text.strip():
        return None

    body = {
        "inputText": text,
        "embeddingConfig": {
            "outputEmbeddingLength": EMBED_DIM
        }
    }

    try:

        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        result = json.loads(response["body"].read())
        embedding = result.get("embedding")

        if not embedding or len(embedding) != EMBED_DIM:
            return None

        return embedding

    except ClientError as err:
        print("Embedding error:", err)
        return None


# ---------------------------------------------------
# Worker — choose what text to embed per item type
# ---------------------------------------------------

def embed_item(item, idx):

    item_type = item["type"]

    try:

        if item_type == "text":
            # Raw text chunk — embed directly
            embed_text = item.get("text", "").strip()

        elif item_type == "table":
            # Prefer Nova summary if available, fall back to raw table text
            embed_text = item.get("summary") or item.get("text", "")
            embed_text = embed_text.strip()

        elif item_type in ("image", "page"):
            # Must have a summary — raw base64 is not embeddable as text
            embed_text = item.get("summary", "").strip()

            if not embed_text:
                print(f"  ⚠ No summary for {item_type} at index {idx} — skipping")
                return None

        else:
            print(f"  ⚠ Unknown item type '{item_type}' at index {idx} — skipping")
            return None

        if not embed_text:
            print(f"  ⚠ Empty embed text for {item_type} at index {idx} — skipping")
            return None

        embedding = generate_text_embedding(embed_text)

        if embedding is None:
            print(f"  ⚠ Titan returned no embedding for index {idx}")
            return None

        item["embedding"] = embedding

        # Store what text was actually embedded so it's inspectable later
        item["embedded_text"] = embed_text

        return item

    except Exception as e:
        print(f"  ❌ Worker crash at index {idx}: {e}")
        return None


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def main():

    print("\nSTEP 2 — Summarization + Embedding Pipeline\n")

    if not os.path.exists(config.CHUNKS_PATH):
        print("❌ Processed items file not found.")
        return

    print("Loading processed items...")

    with open(config.CHUNKS_PATH, "r") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items")

    # --------------------------------------------------
    # PHASE 1 — Generate summaries for images and tables
    # --------------------------------------------------

    summarizer = Summarizer()
    items = summarizer.summarize_items(items)

    # Save summarized items (useful for inspection/debugging)
    summarized_path = os.path.join(config.DATA_DIR, "summarized_items.json")

    with open(summarized_path, "w") as f:
        # Save without base64 image blobs to keep file size manageable
        slim = [
            {k: v for k, v in item.items() if k != "image"}
            for item in items
        ]
        json.dump(slim, f, indent=2)

    print(f"\n✅ Summaries saved → {summarized_path}")

    # --------------------------------------------------
    # PHASE 2 — Embed summaries/text in parallel
    # --------------------------------------------------

    print("\nGenerating embeddings in parallel...\n")

    embedded_items = []

    with ThreadPoolExecutor(max_workers=6) as executor:

        futures = {
            executor.submit(embed_item, item, idx): idx
            for idx, item in enumerate(items)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):

            result = future.result()

            if result is not None:
                embedded_items.append(result)

    if len(embedded_items) == 0:
        raise RuntimeError(
            "No items were embedded — check AWS credentials and Bedrock access."
        )

    print(f"\nValid embeddings: {len(embedded_items)} / {len(items)}")

    # Save — strip image blobs to keep JSON size manageable.
    # The original image data lives on disk; path is stored in each item.
    slim_embedded = [
        {k: v for k, v in item.items() if k != "image"}
        for item in embedded_items
    ]

    with open(config.EMBEDDED_ITEMS_PATH, "w") as f:
        json.dump(slim_embedded, f)

    print(f"\n✅ Embeddings complete → {config.EMBEDDED_ITEMS_PATH}")


# ---------------------------------------------------

if __name__ == "__main__":
    main()