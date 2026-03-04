import os
from tqdm import tqdm
import tabula
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from table_extractor import TableExtractor
from image_extractor import ImageExtractor


class DocumentProcessor:
    """
    Orchestrates multi-modal extraction from a PDF:
      - Tables  -> TableExtractor  (tabula detection + pdfplumber reparsing)
      - Images  -> ImageExtractor  (embedded XObjects + vector drawn regions)
      - Text    -> chunked, skipping regions claimed by tables/images
      - Pages   -> full-page snapshots for scanned PDFs

    Extraction order matters — tables and images run first and register
    their bounding boxes in _claimed_rects so the text extractor skips
    those regions and avoids duplication.
    """

    def __init__(self, pdf_path, base_dir="data"):
        self.pdf_path = pdf_path
        self.base_dir = base_dir
        self.doc      = fitz.open(pdf_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len
        )

        self.items = []

        # Shared registry: page_num -> [fitz.Rect]
        # Tables and images register here; text extractor skips these areas.
        self._claimed_rects = {}

        self._create_directories()

        self._table_extractor = TableExtractor(
            pdf_path=self.pdf_path,
            base_dir=self.base_dir,
            claimed_rects=self._claimed_rects
        )

        self._image_extractor = ImageExtractor(
            pdf_doc=self.doc,
            pdf_path=self.pdf_path,
            base_dir=self.base_dir,
            claimed_rects=self._claimed_rects
        )

    # --------------------------------------------------
    # Directory setup
    # --------------------------------------------------

    def _create_directories(self):
        for d in ["images", "text", "tables", "page_images"]:
            os.makedirs(os.path.join(self.base_dir, d), exist_ok=True)

    # --------------------------------------------------
    # Text blocks (used by text extraction + image captions)
    # --------------------------------------------------

    def _get_text_blocks(self, page):
        blocks = []
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_text    = ""
            max_font_size = 0.0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text    += span.get("text", "")
                    max_font_size  = max(max_font_size, span.get("size", 0))
            block_text = block_text.strip()
            if block_text:
                blocks.append({
                    "text":      block_text,
                    "rect":      fitz.Rect(block["bbox"]),
                    "font_size": max_font_size
                })

        return blocks

    # --------------------------------------------------
    # Text extraction — skips claimed regions
    # --------------------------------------------------

    def _process_text(self, page, page_num):
        """
        Extracts text from the page, skipping any block that significantly
        overlaps a region already claimed by a table or image extractor.
        """
        claimed = self._claimed_rects.get(page_num, [])
        raw     = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        parts   = []

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_rect = fitz.Rect(block["bbox"])
            skip = False

            for cr in claimed:
                inter = block_rect & cr
                if not inter.is_empty:
                    ratio = (inter.width * inter.height) / \
                            (block_rect.width * block_rect.height + 1e-6)
                    if ratio > 0.3:
                        skip = True
                        break
            if skip:
                continue

            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")

            block_text = block_text.strip()
            if block_text:
                parts.append(block_text)

        if not parts:
            return []

        items = []
        for i, chunk in enumerate(self.text_splitter.split_text("\n".join(parts))):
            chunk = chunk.strip()
            if not chunk:
                continue

            file_name = os.path.join(
                self.base_dir, "text",
                f"{os.path.basename(self.pdf_path)}_text_{page_num}_{i}.txt"
            )
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(chunk)

            items.append({
                "type": "text",
                "page": page_num,
                "text": chunk,
                "path": file_name
            })

        return items

    # --------------------------------------------------
    # Main pipeline
    # --------------------------------------------------

    def process_document(self):
        print(f"\nProcessing: {self.pdf_path}\n")

        for page_num in tqdm(range(len(self.doc)), desc="Processing PDF"):
            page        = self.doc[page_num]
            text_blocks = self._get_text_blocks(page)

            # Tables first — claim their rects before text runs
            self.items += self._table_extractor.process_page(page_num)

            # Embedded images — claim their rects
            self.items += self._image_extractor.process_embedded(page, page_num, text_blocks)

            # Drawn/vector regions — skips anything already claimed
            self.items += self._image_extractor.process_drawn(page, page_num, text_blocks)

            # Text last — skips all claimed regions
            self.items += self._process_text(page, page_num)

            # Full page snapshot
            self.items += self._image_extractor.process_page_image(page, page_num)

        self._print_summary()
        return self.items

    def _print_summary(self):
        counts = {}
        for item in self.items:
            counts[item["type"]] = counts.get(item["type"], 0) + 1
        print(f"\n{'='*40}")
        print(f"Extraction complete — {len(self.items)} total items")
        for t, n in sorted(counts.items()):
            print(f"  {t:<15} {n}")
        print(f"{'='*40}\n")

    def close(self):
        self.doc.close()


if __name__ == "__main__":
    import config
    processor = DocumentProcessor(config.PDF_PATH, base_dir=config.DATA_DIR)
    items = processor.process_document()
    print("Sample items:")
    for item in items[:3]:
        preview = {k: v[:80] if isinstance(v, str) and len(v) > 80 else v
                   for k, v in item.items() if k != "image"}
        print(preview)
    processor.close()