import os
import base64
from tqdm import tqdm
import fitz  # PyMuPDF
import tabula
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:

    def __init__(self, pdf_path, base_dir="data"):

        self.pdf_path = pdf_path
        self.base_dir = base_dir
        self.doc = fitz.open(pdf_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len
        )

        self.items = []

        self._create_directories()

    # --------------------------------------------------
    # Directory setup
    # --------------------------------------------------

    def _create_directories(self):

        dirs = ["images", "text", "tables", "page_images"]

        for d in dirs:
            os.makedirs(os.path.join(self.base_dir, d), exist_ok=True)

    # --------------------------------------------------
    # Table extraction (skip empty tables)
    # --------------------------------------------------

    def _process_tables(self, page_num):

        try:

            tables = tabula.read_pdf(
                self.pdf_path,
                pages=page_num + 1,
                multiple_tables=True
            )

            if not tables:
                return

            for idx, table in enumerate(tables):

                table_text = "\n".join(
                    [" | ".join(map(str, row)) for row in table.values]
                ).strip()

                # 🚨 Skip empty tables
                if not table_text:
                    continue

                file_name = os.path.join(
                    self.base_dir,
                    "tables",
                    f"{os.path.basename(self.pdf_path)}_table_{page_num}_{idx}.txt"
                )

                with open(file_name, "w") as f:
                    f.write(table_text)

                self.items.append({
                    "type": "table",
                    "page": page_num,
                    "text": table_text,
                    "path": file_name
                })

        except Exception as e:
            print(f"Table extraction failed page {page_num}: {e}")

    # --------------------------------------------------
    # Text chunking (skip empty chunks)
    # --------------------------------------------------

    def _process_text(self, text, page_num):

        chunks = self.text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):

            chunk = chunk.strip()

            # 🚨 Skip empty text
            if not chunk:
                continue

            file_name = os.path.join(
                self.base_dir,
                "text",
                f"{os.path.basename(self.pdf_path)}_text_{page_num}_{i}.txt"
            )

            with open(file_name, "w") as f:
                f.write(chunk)

            self.items.append({
                "type": "text",
                "page": page_num,
                "text": chunk,
                "path": file_name
            })

    # --------------------------------------------------
    # Embedded images (skip empty files)
    # --------------------------------------------------

    def _process_images(self, page, page_num):

        images = page.get_images()

        for idx, img in enumerate(images):

            xref = img[0]
            pix = fitz.Pixmap(self.doc, xref)

            file_name = os.path.join(
                self.base_dir,
                "images",
                f"{os.path.basename(self.pdf_path)}_image_{page_num}_{idx}_{xref}.png"
            )

            pix.save(file_name)

            if os.path.getsize(file_name) == 0:
                continue

            with open(file_name, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf8")

            self.items.append({
                "type": "image",
                "page": page_num,
                "image": encoded,
                "path": file_name
            })

    # --------------------------------------------------
    # Full page snapshot
    # --------------------------------------------------

    def _process_page_image(self, page, page_num):

        pix = page.get_pixmap()

        file_name = os.path.join(
            self.base_dir,
            "page_images",
            f"page_{page_num:03d}.png"
        )

        pix.save(file_name)

        if os.path.getsize(file_name) == 0:
            return

        with open(file_name, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf8")

        self.items.append({
            "type": "page",
            "page": page_num,
            "image": encoded,
            "path": file_name
        })

    # --------------------------------------------------
    # Main pipeline
    # --------------------------------------------------

    def process_document(self):

        print(f"\nProcessing: {self.pdf_path}\n")

        for page_num in tqdm(range(len(self.doc)), desc="Processing PDF"):

            page = self.doc[page_num]

            text = page.get_text()

            if text.strip():
                self._process_text(text, page_num)

            self._process_tables(page_num)
            self._process_images(page, page_num)
            self._process_page_image(page, page_num)

        print(f"\nTotal extracted items: {len(self.items)}")

        return self.items

    # --------------------------------------------------

    def close(self):

        self.doc.close()


# ------------------------------------------------------
# Standalone test
# ------------------------------------------------------

if __name__ == "__main__":

    processor = DocumentProcessor(
        "/Users/paulimmanuel/Desktop/multi-model_assignment 2/data/raw/qatar_test_doc.pdf"
    )

    items = processor.process_document()

    print("\nSample item:")
    print(items[0])

    processor.close()
