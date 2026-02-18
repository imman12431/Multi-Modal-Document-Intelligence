import os

# --------------------------------------------------
# Base paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Notebook-style multimodal folders
TEXT_DIR = os.path.join(DATA_DIR, "text")
TABLES_DIR = os.path.join(DATA_DIR, "tables")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")

# --------------------------------------------------
# Files used in pipeline
# --------------------------------------------------

# ✅ Matches your actual PDF location
PDF_PATH = os.path.join(RAW_DIR, "qatar_test_doc.pdf")

# JSON outputs
CHUNKS_PATH = os.path.join(DATA_DIR, "extracted_items.json")
EMBEDDED_ITEMS_PATH = os.path.join(DATA_DIR, "embedded_items.json")

# --------------------------------------------------
# Models (if needed elsewhere)
# --------------------------------------------------

EMBEDDING_MODEL = "amazon.titan-embed-image-v1"
LLM_MODEL = "amazon.nova-pro-v1"

# --------------------------------------------------
# Directory creator
# --------------------------------------------------

def create_directories():

    dirs = [
        DATA_DIR,
        RAW_DIR,
        TEXT_DIR,
        TABLES_DIR,
        IMAGES_DIR,
        PAGE_IMAGES_DIR,
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("✓ Multimodal data directories ready")


# --------------------------------------------------

if __name__ == "__main__":

    create_directories()

    print("\nDirectory layout:")
    print("PDF location:", PDF_PATH)
    print("Text:", TEXT_DIR)
    print("Tables:", TABLES_DIR)
    print("Images:", IMAGES_DIR)
    print("Page images:", PAGE_IMAGES_DIR)
