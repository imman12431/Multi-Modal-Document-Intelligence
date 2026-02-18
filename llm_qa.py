import json
from langchain_aws import ChatBedrock


class NovaMultimodalQA:

    # --------------------------------------------------
    # Init Nova client
    # --------------------------------------------------

    def __init__(self, region="us-east-1"):

        print("Initializing Amazon Nova multimodal QA...")

        self.model_id = "nvidia.nemotron-nano-12b-v2"

        self.client = ChatBedrock(
            model_id=self.model_id,
            region_name=region
        )

        print("Nova client ready.")

    # --------------------------------------------------
    # Build Nova request payload
    # --------------------------------------------------

    def _build_request(self, prompt, matched_items):

        system_msg = [
            {
                "text": (
                    "You are a helpful assistant for question answering.\n"
                    "The provided text and images are retrieved context.\n"
                    "Use them to answer accurately."
                )
            }
        ]

        message_content = []

        for item in matched_items:

            item_type = item.get("type")

            if item_type in ["text", "table"]:

                text = item.get("text", "").strip()

                if text:
                    message_content.append({"text": text})

            else:

                image_data = item.get("image")

                if image_data:
                    message_content.append({
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_data}
                        }
                    })

        # user question appended last
        message_list = [
            {"role": "user", "content": message_content},
            {"role": "user", "content": [{"text": prompt}]}
        ]

        inference_params = {
            "max_new_tokens": 300,
            "top_p": 0.9,
            "top_k": 20
        }

        return {
            "messages": message_list,
            "system": system_msg,
            "inferenceConfig": inference_params
        }

    # --------------------------------------------------
    # Main QA call
    # --------------------------------------------------

    def generate_answer(self, prompt, matched_items):

        if not matched_items:
            return "No relevant context retrieved."

        request_payload = self._build_request(prompt, matched_items)

        try:

            response = self.client.invoke(
                json.dumps(request_payload)
            )

            return response.content


        except Exception as e:

            import traceback

            print("\n🚨 NOVA ERROR")

            print("Exception:", e)

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

    # mock example retrieved items
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
    print("\nContext metadata:", result["metadata"])
