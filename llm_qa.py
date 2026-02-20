import os
import json
import base64
import traceback
import boto3
from botocore.exceptions import ClientError


class NovaMultimodalQA:

    # --------------------------------------------------
    # Init
    # --------------------------------------------------

    def __init__(self):

        self.model_id = "amazon.nova-pro-v1:0"

        region = os.getenv("AWS_REGION", "us-east-1")

        print(f"Initializing Nova QA — model: {self.model_id}, region: {region}")
        print(f"AWS key present: {bool(os.getenv('AWS_ACCESS_KEY_ID'))}")

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )

        print("✅ Nova QA initialized")

    # --------------------------------------------------
    # Load original image from disk
    # --------------------------------------------------

    def _load_image(self, item):
        """
        Retrieves the original image for an image/page item.
        Checks for in-memory base64 first (if still in item),
        then falls back to loading from the path on disk.
        """

        # If image is still in memory (e.g. during testing)
        if "image" in item:
            return item["image"]

        path = item.get("path", "")

        if not path or not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # --------------------------------------------------
    # Build Nova messages with multimodal context
    # --------------------------------------------------

    def _build_messages(self, prompt, matched_items):
        """
        Builds the Nova message payload.

        - Text and table items → added as text blocks in the system context
        - Image and page items → loaded from disk and added as inline images
          so Nova sees the actual visual content, not just the summary

        The summary is also included as a label alongside each image so
        Nova has a semantic anchor for what it's looking at.
        """

        system_text = (
            "You are a helpful assistant answering questions using retrieved context from a PDF document. "
            "The context below includes text passages, table data, and images. "
            "Use all provided context to give a thorough, accurate answer. "
            "If a piece of context is not relevant to the question, ignore it."
        )

        # User message content — interleaved text and images
        user_content = []

        text_context_parts = []
        image_count = 0

        for item in matched_items:

            item_type = item.get("type")
            page = item.get("page")
            page_label = f"(page {page + 1})" if page is not None else ""

            if item_type == "text":

                text = item.get("text", "").strip()
                if text:
                    text_context_parts.append(f"[Text {page_label}]\n{text}")

            elif item_type == "table":

                # Use summary for readability, append raw data for precision
                summary = item.get("summary", "")
                raw = item.get("text", "")

                if summary:
                    text_context_parts.append(
                        f"[Table {page_label} — Summary]\n{summary}"
                    )
                if raw:
                    text_context_parts.append(
                        f"[Table {page_label} — Raw Data]\n{raw}"
                    )

            elif item_type in ("image", "page"):

                image_b64 = self._load_image(item)

                if not image_b64:
                    # Fall back to summary only if image can't be loaded
                    summary = item.get("summary", "")
                    if summary:
                        text_context_parts.append(
                            f"[Image {page_label} — Summary only, image unavailable]\n{summary}"
                        )
                    continue

                image_count += 1
                summary = item.get("summary", "")
                caption = item.get("caption", "")

                # Add a text label before the image so Nova has context
                label_parts = [f"[Image {image_count} {page_label}]"]
                if caption:
                    label_parts.append(f"Caption: {caption}")
                if summary:
                    label_parts.append(f"Summary: {summary}")

                user_content.append({
                    "text": "\n".join(label_parts)
                })

                user_content.append({
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": image_b64
                        }
                    }
                })

        # Prepend all text context as a single block
        if text_context_parts:
            user_content.insert(0, {
                "text": "--- Retrieved Context ---\n\n" + "\n\n".join(text_context_parts)
            })

        # Finally append the actual question
        user_content.append({
            "text": f"\n--- Question ---\n{prompt}"
        })

        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]

        print(f"\n=== Nova context: {len(matched_items)} items "
              f"({image_count} images, "
              f"{len(text_context_parts)} text/table blocks) ===\n")

        return system_text, messages

    # --------------------------------------------------
    # Main QA call
    # --------------------------------------------------

    def generate_answer(self, prompt, matched_items):

        if not matched_items:
            return "No relevant context retrieved."

        system_text, messages = self._build_messages(prompt, matched_items)

        body = {
            "system": [{"text": system_text}],
            "messages": messages,
            "inferenceConfig": {
                "max_new_tokens": 1024,
                "temperature": 0.3
            }
        }

        try:

            print("Calling Nova...")

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )

            result = json.loads(response["body"].read())

            return result["output"]["message"]["content"][0]["text"].strip()

        except ClientError as e:
            print(f"\n🚨 Nova ClientError: {e}")
            traceback.print_exc()
            return f"Nova error: {str(e)}"

        except Exception as e:
            print(f"\n🚨 Nova unexpected error: {e}")
            traceback.print_exc()
            return f"Nova error: {str(e)}"

    # --------------------------------------------------
    # QA with metadata output
    # --------------------------------------------------

    def generate_answer_with_context(self, prompt, matched_items):

        answer = self.generate_answer(prompt, matched_items)

        metadata = [
            {
                "type": item.get("type"),
                "page": item.get("page"),
                "path": item.get("path"),
                "summary": item.get("summary", "")[:120] + "..."
                           if item.get("summary") else None
            }
            for item in matched_items
        ]

        return {
            "answer": answer,
            "context_items": len(matched_items),
            "metadata": metadata
        }


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":

    print("\n=== Standalone Nova QA test ===")

    test_items = [
        {
            "type": "text",
            "page": 0,
            "text": "Qatar's GDP grew by 4.2% in 2024, driven by LNG exports.",
            "path": "sample.txt"
        },
        {
            "type": "table",
            "page": 1,
            "text": "Year | GDP Growth\n2022 | 3.1%\n2023 | 3.8%\n2024 | 4.2%",
            "summary": "Table showing Qatar GDP growth from 2022 to 2024, rising steadily to 4.2%.",
            "path": "sample_table.txt"
        }
    ]

    qa = NovaMultimodalQA()

    result = qa.generate_answer_with_context(
        "What was Qatar's GDP growth in 2024?",
        test_items
    )

    print("\nAnswer:\n", result["answer"])
    print("\nMetadata:", result["metadata"])