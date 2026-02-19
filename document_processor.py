import os
import base64
from tqdm import tqdm
import fitz  # PyMuPDF
import tabula
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Thresholds — tune these for your PDFs
# --------------------------------------------------

# Minimum image area (px²) to bother keeping
MIN_IMAGE_AREA = 5_000

# Font size above which a line is treated as a heading/caption
CAPTION_FONT_SIZE_THRESHOLD = 9.0

# How close (pts) a caption block must be to an image rect to be linked
CAPTION_PROXIMITY_PTS = 40


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
        self._seen_image_xrefs = set()   # deduplicate shared XObjects

        self._create_directories()

    # --------------------------------------------------
    # Directory setup
    # --------------------------------------------------

    def _create_directories(self):

        dirs = ["images", "text", "tables", "page_images"]

        for d in dirs:
            os.makedirs(os.path.join(self.base_dir, d), exist_ok=True)

    # --------------------------------------------------
    # Safe Pixmap save — handles CMYK and alpha
    # --------------------------------------------------

    def _save_pixmap(self, pix, file_name):
        """Convert to RGB if needed, then save. Returns False if empty."""

        if pix.n > 4:
            # CMYK or exotic colorspace → convert to RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.alpha:
            # Drop alpha channel
            pix = fitz.Pixmap(fitz.csRGB, pix)

        pix.save(file_name)

        if os.path.getsize(file_name) == 0:
            return False

        return True

    # --------------------------------------------------
    # Encode image file to base64
    # --------------------------------------------------

    def _encode_image(self, file_name):

        with open(file_name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf8")

    # --------------------------------------------------
    # Extract text blocks with layout info
    # --------------------------------------------------

    def _get_text_blocks(self, page):
        """
        Returns list of dicts with text + bounding rect.
        Uses 'dict' mode so we get per-span font sizes.
        """

        blocks = []

        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in raw.get("blocks", []):

            if block.get("type") != 0:   # 0 = text block
                continue

            block_text = ""
            max_font_size = 0.0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                    max_font_size = max(max_font_size, span.get("size", 0))

            block_text = block_text.strip()

            if not block_text:
                continue

            blocks.append({
                "text": block_text,
                "rect": fitz.Rect(block["bbox"]),
                "font_size": max_font_size
            })

        return blocks

    # --------------------------------------------------
    # Find caption text near an image rect
    # --------------------------------------------------

    def _find_caption(self, image_rect, text_blocks):
        """
        Look for a text block directly above or below the image.
        Returns the caption string or empty string.
        """

        candidates = []

        for block in text_blocks:

            r = block["rect"]

            # Must horizontally overlap with the image
            h_overlap = (
                r.x0 < image_rect.x1 and
                r.x1 > image_rect.x0
            )

            if not h_overlap:
                continue

            # Distance above the image (caption above)
            dist_above = image_rect.y0 - r.y1
            # Distance below the image (caption below)
            dist_below = r.y0 - image_rect.y1

            if 0 <= dist_above <= CAPTION_PROXIMITY_PTS:
                candidates.append((dist_above, block["text"]))

            elif 0 <= dist_below <= CAPTION_PROXIMITY_PTS:
                candidates.append((dist_below, block["text"]))

        if not candidates:
            return ""

        # Pick closest
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # --------------------------------------------------
    # TEXT — chunk and save
    # --------------------------------------------------

    def _process_text(self, text, page_num):

        chunks = self.text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):

            chunk = chunk.strip()

            if not chunk:
                continue

            file_name = os.path.join(
                self.base_dir,
                "text",
                f"{os.path.basename(self.pdf_path)}_text_{page_num}_{i}.txt"
            )

            with open(file_name, "w", encoding="utf-8") as f:
                f.write(chunk)

            self.items.append({
                "type": "text",
                "page": page_num,
                "text": chunk,
                "path": file_name
            })

    # --------------------------------------------------
    # TABLES — text-layer tables via tabula
    # --------------------------------------------------

    def _process_tables_tabula(self, page_num):
        """
        Use tabula for text-layer tables. Skips silently if tabula
        finds nothing (image-based tables are caught by _process_images).
        """

        try:

            tables = tabula.read_pdf(
                self.pdf_path,
                pages=page_num + 1,
                multiple_tables=True,
                silent=True
            )

            if not tables:
                return

            for idx, table in enumerate(tables):

                if table.empty:
                    continue

                # Replace NaN with empty string for clean output
                table = table.fillna("")

                table_text = table.to_csv(index=False, sep="|").strip()

                if not table_text:
                    continue

                file_name = os.path.join(
                    self.base_dir,
                    "tables",
                    f"{os.path.basename(self.pdf_path)}_table_{page_num}_{idx}.txt"
                )

                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(table_text)

                self.items.append({
                    "type": "table",
                    "page": page_num,
                    "text": table_text,
                    "path": file_name
                })

        except Exception as e:
            print(f"  ⚠ Tabula failed page {page_num}: {e}")

    # --------------------------------------------------
    # IMAGES — embedded XObjects (e.g. photos, logos)
    # --------------------------------------------------

    def _process_embedded_images(self, page, page_num, text_blocks):
        """
        Extracts discrete image objects embedded in the PDF's object tree.
        Deduplicates by xref so shared images (headers, logos) aren't
        repeated on every page.
        """

        for idx, img in enumerate(page.get_images(full=True)):

            xref = img[0]

            # Skip images we've already seen (shared XObjects)
            if xref in self._seen_image_xrefs:
                continue

            self._seen_image_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(self.doc, xref)
            except Exception as e:
                print(f"  ⚠ Could not read image xref={xref}: {e}")
                continue

            # Skip tiny images (icons, bullets, spacers)
            if pix.width * pix.height < MIN_IMAGE_AREA:
                continue

            file_name = os.path.join(
                self.base_dir,
                "images",
                f"{os.path.basename(self.pdf_path)}_xobj_{page_num}_{idx}_{xref}.png"
            )

            if not self._save_pixmap(pix, file_name):
                continue

            encoded = self._encode_image(file_name)

            # Try to locate the image on the page to find a caption
            image_rects = page.get_image_rects(xref)
            caption = ""
            if image_rects:
                caption = self._find_caption(image_rects[0], text_blocks)

            self.items.append({
                "type": "image",
                "page": page_num,
                "caption": caption,
                "image": encoded,
                "path": file_name
            })

    # --------------------------------------------------
    # IMAGES — vector/drawn regions (charts, image-tables)
    # --------------------------------------------------

    def _process_drawn_regions(self, page, page_num, text_blocks):
        """
        Detects image-like regions painted with PDF drawing operators —
        these are invisible to get_images() but show up in the page's
        drawing commands. Crops and saves each region.

        This is the key fix for tables/charts rendered as vector graphics
        or rasterized into the page stream rather than stored as XObjects.
        """

        # get_drawings() returns all filled/stroked paths on the page
        drawings = page.get_drawings()

        if not drawings:
            return

        # Cluster drawing rects into contiguous regions.
        # A "region" is a group of drawings whose bounding boxes are close
        # enough together that they likely form a single visual element.
        regions = _cluster_rects(
            [fitz.Rect(d["rect"]) for d in drawings],
            gap_threshold=20
        )

        page_rect = page.rect

        for region_idx, region_rect in enumerate(regions):

            # Skip regions that are nearly the full page (background fills)
            region_area = region_rect.width * region_rect.height
            page_area = page_rect.width * page_rect.height

            if region_area > page_area * 0.85:
                continue

            # Skip tiny regions
            if region_area < MIN_IMAGE_AREA:
                continue

            # Expand slightly for padding
            clip = region_rect + (-4, -4, 4, 4)
            clip = clip & page_rect   # clamp to page

            # Render at 2x resolution for sharpness
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=clip)

            if pix.width * pix.height < MIN_IMAGE_AREA:
                continue

            file_name = os.path.join(
                self.base_dir,
                "images",
                f"{os.path.basename(self.pdf_path)}_region_{page_num}_{region_idx}.png"
            )

            if not self._save_pixmap(pix, file_name):
                continue

            encoded = self._encode_image(file_name)
            caption = self._find_caption(region_rect, text_blocks)

            self.items.append({
                "type": "image",
                "page": page_num,
                "caption": caption,
                "image": encoded,
                "path": file_name
            })

    # --------------------------------------------------
    # FULL PAGE SNAPSHOT — fallback for scanned PDFs
    # --------------------------------------------------

    def _process_page_image(self, page, page_num):
        """
        Renders the full page as a high-res image.
        Used as a fallback so scanned PDFs without any text layer
        still get their content into the pipeline.
        """

        # 2x resolution matrix
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)

        file_name = os.path.join(
            self.base_dir,
            "page_images",
            f"page_{page_num:03d}.png"
        )

        if not self._save_pixmap(pix, file_name):
            return

        encoded = self._encode_image(file_name)

        self.items.append({
            "type": "page",
            "page": page_num,
            "caption": f"Full page snapshot — page {page_num + 1}",
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

            # Get layout-aware text blocks for caption detection
            text_blocks = self._get_text_blocks(page)

            # --- Text ---
            raw_text = page.get_text()
            if raw_text.strip():
                self._process_text(raw_text, page_num)

            # --- Tables (text layer only via tabula) ---
            self._process_tables_tabula(page_num)

            # --- Embedded images (XObjects) ---
            self._process_embedded_images(page, page_num, text_blocks)

            # --- Drawn/vector regions (charts, image-tables) ---
            self._process_drawn_regions(page, page_num, text_blocks)

            # --- Full page snapshot ---
            self._process_page_image(page, page_num)

        self._print_summary()

        return self.items

    # --------------------------------------------------

    def _print_summary(self):

        counts = {}
        for item in self.items:
            t = item["type"]
            counts[t] = counts.get(t, 0) + 1

        print(f"\n{'='*40}")
        print(f"Extraction complete — {len(self.items)} total items")
        for t, n in sorted(counts.items()):
            print(f"  {t:<15} {n}")
        print(f"{'='*40}\n")

    # --------------------------------------------------

    def close(self):
        self.doc.close()


# --------------------------------------------------
# Rect clustering helper (module-level)
# --------------------------------------------------

def _cluster_rects(rects, gap_threshold=20):
    """
    Groups a list of fitz.Rect objects into clusters where any two
    rects in the same cluster are within gap_threshold points of each other.
    Returns one merged bounding rect per cluster.
    """

    if not rects:
        return []

    clusters = []   # list of lists of rects

    for rect in rects:

        merged = False

        for cluster in clusters:

            # Check if rect is close to any rect already in this cluster
            for existing in cluster:

                dx = max(0, max(existing.x0, rect.x0) - min(existing.x1, rect.x1))
                dy = max(0, max(existing.y0, rect.y0) - min(existing.y1, rect.y1))

                if dx <= gap_threshold and dy <= gap_threshold:
                    cluster.append(rect)
                    merged = True
                    break

            if merged:
                break

        if not merged:
            clusters.append([rect])

    # Merge each cluster into a single bounding rect
    merged_rects = []

    for cluster in clusters:
        x0 = min(r.x0 for r in cluster)
        y0 = min(r.y0 for r in cluster)
        x1 = max(r.x1 for r in cluster)
        y1 = max(r.y1 for r in cluster)
        merged_rects.append(fitz.Rect(x0, y0, x1, y1))

    return merged_rects


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":

    import config

    processor = DocumentProcessor(
        config.PDF_PATH,
        base_dir=config.DATA_DIR
    )

    items = processor.process_document()

    print("Sample items:")
    for item in items[:3]:
        preview = {k: v[:80] if isinstance(v, str) and len(v) > 80 else v
                   for k, v in item.items() if k != "image"}
        print(preview)

    processor.close()