import os
import json
import time
import hashlib
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from botocore.exceptions import ClientError


# --------------------------------------------------
# Backend switch
# Set to "nova" for production, "ollama" for local dev
# --------------------------------------------------

BACKEND = os.getenv("SUMMARIZER_BACKEND", "nova")  # "nova" | "ollama"

# --------------------------------------------------
# Nova settings
# --------------------------------------------------

NOVA_MODEL_ID  = "amazon.nova-pro-v1:0"
NOVA_REGION    = os.getenv("AWS_REGION", "us-east-1")

# --------------------------------------------------
# Ollama settings
# --------------------------------------------------

OLLAMA_URL         = "http://localhost:11434/api/generate"
OLLAMA_IMAGE_MODEL = "llava"     # vision-capable
OLLAMA_TABLE_MODEL = "mistral"   # text-only, faster

# --------------------------------------------------
# Cache — persisted to disk so re-runs skip Nova calls
# --------------------------------------------------

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "summary_cache.json"
)

# --------------------------------------------------
# Prompts
# --------------------------------------------------

IMAGE_SUMMARY_PROMPT = (
    "You are analyzing an image extracted from a PDF document. "
    "Describe everything you can see in this image in detail: "
    "any text, numbers, labels, trends, structure, or visual content. "
    "If it appears to be a chart or graph, describe the axes, data, and key takeaways. "
    "If it appears to be a table rendered as an image, transcribe its contents as accurately as possible. "
    "Be specific and thorough — this summary will be used to match the image to user questions."
)

TABLE_SUMMARY_PROMPT = (
    "You are analyzing a data table extracted from a PDF document. "
    "Write a concise but thorough natural language summary of this table: "
    "describe what it measures, the column/row structure, key values, and any notable patterns or conclusions. "
    "Do not just repeat the raw data — explain what it means."
)


class Summarizer:

    # --------------------------------------------------
    # Init
    # --------------------------------------------------

    def __init__(self, backend=BACKEND, region=NOVA_REGION):

        self.backend = backend

        if backend == "nova":
            self.nova_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region
            )
            print(f"✅ Summarizer initialized — backend: Nova ({NOVA_MODEL_ID})")

        elif backend == "ollama":
            # Quick connectivity check
            try:
                requests.get("http://localhost:11434", timeout=3)
                print(f"✅ Summarizer initialized — backend: Ollama "
                      f"(images: {OLLAMA_IMAGE_MODEL}, tables: {OLLAMA_TABLE_MODEL})")
            except Exception:
                raise RuntimeError(
                    "Ollama backend selected but server not reachable at localhost:11434. "
                    "Run: ollama serve"
                )

        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'nova' or 'ollama'.")

        # Load cache
        self._cache      = self._load_cache()
        self._cache_lock = threading.Lock()

        print(f"   Cache: {len(self._cache)} existing summaries loaded from {CACHE_PATH}")

    # --------------------------------------------------
    # Cache helpers
    # --------------------------------------------------

    def _load_cache(self):
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                print("  ⚠ Cache file corrupted — starting fresh")
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with self._cache_lock:
            with open(CACHE_PATH, "w") as f:
                json.dump(self._cache, f)

    def _cache_key(self, item):
        """
        Stable hash of the item's raw content.
        Images  → hash of base64 bytes (the actual pixels).
        Tables  → hash of raw table text.
        Survives re-runs even if page numbers or paths change.
        """
        item_type = item["type"]
        content   = item.get("image", "") if item_type == "image" else item.get("text", "")
        return hashlib.sha256(content.encode()).hexdigest()

    # --------------------------------------------------
    # Public: summarize a single image item
    # --------------------------------------------------

    def summarize_image(self, image_b64, caption="", page=None):

        prompt = IMAGE_SUMMARY_PROMPT

        if caption:
            prompt += f"\n\nCaption hint (if available): {caption}"

        if page is not None:
            prompt += f"\nThis image is from page {page + 1} of the document."

        if self.backend == "nova":
            return self._call_nova(
                user_content=[
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_b64}
                        }
                    },
                    {"text": prompt}
                ]
            )

        else:  # ollama
            return self._call_ollama(
                prompt=prompt,
                model=OLLAMA_IMAGE_MODEL,
                image_b64=image_b64
            )

    # --------------------------------------------------
    # Public: summarize a single table item
    # --------------------------------------------------

    def summarize_table(self, table_text, page=None):

        prompt = TABLE_SUMMARY_PROMPT + f"\n\nTable data:\n{table_text}"

        if page is not None:
            prompt += f"\n\nThis table is from page {page + 1} of the document."

        if self.backend == "nova":
            return self._call_nova(
                user_content=[{"text": prompt}]
            )

        else:  # ollama
            return self._call_ollama(
                prompt=prompt,
                model=OLLAMA_TABLE_MODEL
            )

    # --------------------------------------------------
    # Nova backend
    # --------------------------------------------------

    def _call_nova(self, user_content):

        body = {
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "inferenceConfig": {
                "max_new_tokens": 512,
                "temperature": 0.2
            }
        }

        response = self.nova_client.invoke_model(
            modelId=NOVA_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        result = json.loads(response["body"].read())
        return result["output"]["message"]["content"][0]["text"].strip()

    # --------------------------------------------------
    # Ollama backend
    # --------------------------------------------------

    def _call_ollama(self, prompt, model, image_b64=None):

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        if image_b64:
            payload["images"] = [image_b64]

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        return response.json().get("response", "").strip()

    # --------------------------------------------------
    # Batch summarize — parallel, cached, with retry
    # --------------------------------------------------

    def summarize_items(self, items, max_workers=6, max_retries=3):
        """
        Summarizes all image and table items in parallel.

        What's skipped:
          - text items  (already readable, no summary needed)
          - page items  (full-page snapshots redundant with other items)
          - cache hits  (no API call made at all)

        max_workers : concurrent API calls (5-8 safe for Bedrock limits)
        max_retries : retries on throttling with exponential backoff
        """

        needs_summary = [
            i for i, item in enumerate(items)
            if item["type"] in ("image", "table")   # page intentionally excluded
        ]

        # Split into cache hits vs items needing a real API call
        cache_hits = []
        needs_api  = []

        for i in needs_summary:
            key = self._cache_key(items[i])
            if key in self._cache:
                cache_hits.append((i, key))
            else:
                needs_api.append((i, key))

        skipped_text = sum(1 for item in items if item["type"] == "text")
        skipped_page = sum(1 for item in items if item["type"] == "page")

        print(f"\n── Summarization ──────────────────────────────")
        print(f"   Total items       : {len(items)}")
        print(f"   Skipped (text)    : {skipped_text}")
        print(f"   Skipped (page)    : {skipped_page}")
        print(f"   Need summary      : {len(needs_summary)}")
        print(f"   Cache hits        : {len(cache_hits)}  ← no API call")
        print(f"   API calls needed  : {len(needs_api)}")
        print(f"   Backend           : {self.backend}")
        print(f"   Workers           : {max_workers}")
        print(f"───────────────────────────────────────────────\n")

        # Apply cache hits instantly — no API call
        for i, key in cache_hits:
            items[i]["summary"] = self._cache[key]

        if not needs_api:
            print("✅ All summaries served from cache — no API calls made\n")
            return items

        # Parallel API calls for uncached items
        completed = 0
        lock      = threading.Lock()

        def summarize_one(i, key):

            nonlocal completed

            item      = items[i]
            item_type = item["type"]
            page      = item.get("page")
            label     = f"{item_type} — page {page + 1}" if page is not None else item_type

            for attempt in range(1, max_retries + 1):

                try:

                    if item_type == "image":

                        image_b64 = item.get("image", "")

                        if not image_b64:
                            print(f"  ⚠ No image data — {label}, skipping")
                            return

                        summary = self.summarize_image(
                            image_b64=image_b64,
                            caption=item.get("caption", ""),
                            page=page
                        )

                    else:  # table

                        table_text = item.get("text", "")

                        if not table_text:
                            print(f"  ⚠ No table text — {label}, skipping")
                            return

                        summary = self.summarize_table(table_text, page=page)

                    if summary:
                        item["summary"] = summary

                        # Write to in-memory cache (thread-safe)
                        with self._cache_lock:
                            self._cache[key] = summary

                    else:
                        print(f"  ⚠ Empty summary returned — {label}")

                    break  # success — exit retry loop

                except ClientError as e:

                    code = e.response["Error"]["Code"]

                    if code == "ThrottlingException" and attempt < max_retries:
                        wait = 2 ** attempt  # 2s, 4s, 8s
                        print(f"  ⏳ Throttled — {label}, retrying in {wait}s "
                              f"(attempt {attempt}/{max_retries})")
                        time.sleep(wait)

                    else:
                        print(f"  ❌ Failed — {label}: {e}")
                        break

                except Exception as e:
                    print(f"  ❌ Unexpected error — {label}: {e}")
                    traceback.print_exc()
                    break

            with lock:
                completed += 1
                print(f"  [{completed}/{len(needs_api)}] {label}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(summarize_one, i, key)
                for i, key in needs_api
            ]
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    print(f"  ❌ Worker error: {exc}")

        # Persist updated cache to disk after all workers finish
        self._save_cache()

        succeeded = sum(1 for i in needs_summary if "summary" in items[i])
        print(f"\n✅ Summarization complete — "
              f"{succeeded}/{len(needs_summary)} summaries "
              f"({len(cache_hits)} from cache, "
              f"{succeeded - len(cache_hits)} new)\n")

        return items