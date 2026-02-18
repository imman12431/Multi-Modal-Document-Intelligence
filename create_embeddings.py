import os
import json
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import config


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
# Titan embedding call
# ---------------------------------------------------

def generate_multimodal_embeddings(prompt=None, image=None):

    if not prompt and not image:
        return None

    body = {
        "embeddingConfig": {
            "outputEmbeddingLength": EMBED_DIM
        }
    }

    if prompt:
        body["inputText"] = prompt

    if image:
        body["inputImage"] = image

    try:

        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        result = json.loads(response["body"].read())

        embedding = result.get("embedding")

        # sanity check
        if not embedding or len(embedding) != EMBED_DIM:
            return None

        return embedding

    except ClientError as err:

        print("Embedding error:", err)
        return None


# ---------------------------------------------------
# Worker function
# ---------------------------------------------------

def embed_item(item, idx):

    item_type = item["type"]

    try:

        if item_type in ["text", "table"]:

            text = item.get("text", "").strip()

            if not text:
                print(f"⚠ Skipping empty text/table → index {idx}")
                return None

            embedding = generate_multimodal_embeddings(prompt=text)

        else:

            image = item.get("image")

            if not image:
                print(f"⚠ Skipping empty image → index {idx}")
                return None

            embedding = generate_multimodal_embeddings(image=image)

        if embedding is None:

            print(f"⚠ Failed embedding → index {idx}")
            return None

        item["embedding"] = embedding

        return item

    except Exception as e:

        print(f"❌ Worker crash at index {idx}:", e)
        return None


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def main():

    print("\nSTEP 2 — Multimodal Embedding Pipeline\n")

    if not os.path.exists(config.CHUNKS_PATH):

        print("❌ Processed items file not found.")
        return

    print("Loading processed items...")

    with open(config.CHUNKS_PATH, "r") as f:

        items = json.load(f)

    print(f"Loaded {len(items)} items")

    print("\nGenerating embeddings in parallel...\n")

    max_workers = 6

    embedded_items = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(embed_item, item, idx): idx
            for idx, item in enumerate(items)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):

            result = future.result()

            if result is not None:
                embedded_items.append(result)

    print(f"\nValid embeddings: {len(embedded_items)}")

    with open(config.EMBEDDED_ITEMS_PATH, "w") as f:

        json.dump(embedded_items, f)

    print("\n✅ Embeddings complete!")
    print(f"Saved → {config.EMBEDDED_ITEMS_PATH}")


# ---------------------------------------------------

if __name__ == "__main__":
    main()
