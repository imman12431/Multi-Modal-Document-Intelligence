import json
import os
from document_processor import DocumentProcessor
import config


def main():

    print("=" * 70)
    print("STEP 1: Document Processing")
    print("=" * 70)

    config.create_directories()

    if not os.path.exists(config.PDF_PATH):
        print("\nERROR: PDF not found")
        return

    print("\nFound PDF")

    processor = DocumentProcessor(
        config.PDF_PATH,
        base_dir=config.DATA_DIR
    )

    chunks = processor.process_document()
    processor.close()

    print(f"\nExtracted {len(chunks)} items")

    text_count = sum(1 for c in chunks if c["type"] == "text")
    table_count = sum(1 for c in chunks if c["type"] == "table")
    image_count = sum(1 for c in chunks if c["type"] == "image")
    page_count = sum(1 for c in chunks if c["type"] == "page")

    print(f"  - Text chunks: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images: {image_count}")
    print(f"  - Page snapshots: {page_count}")

    print("\nSaving extracted items...")

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
