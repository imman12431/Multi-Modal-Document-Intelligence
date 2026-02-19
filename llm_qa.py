import os
import traceback
from langchain_aws import ChatBedrock


class NovaMultimodalQA:

    # --------------------------------------------------
    # Initialize Nova client
    # --------------------------------------------------

    def __init__(self):

        print("\n=== NOVA INIT DEBUG ===")

        self.model_id = "amazon.nova-pro-v1:0"

        region = os.getenv("AWS_REGION", "us-east-1")

        print("Model:", self.model_id)
        print("Region:", region)
        print("AWS key present:",
              bool(os.getenv("AWS_ACCESS_KEY_ID")))
        print("=======================\n")

        try:

            self.client = ChatBedrock(
                model_id=self.model_id,
                region_name=region
            )

            print("✅ Nova client initialized")

        except Exception as e:

            print("❌ Nova init failure")
            traceback.print_exc()
            raise e

    # --------------------------------------------------
    # Build chat prompt
    # --------------------------------------------------

    def _build_messages(self, prompt, matched_items):

        context_text = ""

        for item in matched_items:

            item_type = item.get("type")

            if item_type in ["text", "table"]:

                text = item.get("text", "").strip()

                if text:
                    context_text += text + "\n\n"

        system_prompt = (
            "You are a helpful assistant answering questions using retrieved context.\n"
            "Use only the provided information when possible.\n\n"
            f"Context:\n{context_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        print("\n=== DEBUG — BUILT PROMPT ===")
        print(system_prompt[:1000])
        print("============================\n")

        return messages

    # --------------------------------------------------
    # Main QA call
    # --------------------------------------------------

    def generate_answer(self, prompt, matched_items):

        if not matched_items:
            return "No relevant context retrieved."

        messages = self._build_messages(prompt, matched_items)

        try:

            print("Calling Nova model...")

            response = self.client.invoke(messages)

            print("Nova response received")

            return response.content

        except Exception as e:

            print("\n🚨 NOVA CALL ERROR")
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
                "path": item.get("path")
            }
            for item in matched_items[:5]
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
            "page": 1,
            "text": "Qatar economy grew strongly in 2024.",
            "path": "sample.txt"
        }
    ]

    qa = NovaMultimodalQA()

    result = qa.generate_answer_with_context(
        "What happened to Qatar’s economy?",
        test_items
    )

    print("\nAnswer:\n", result["answer"])
    print("\nMetadata:", result["metadata"])
