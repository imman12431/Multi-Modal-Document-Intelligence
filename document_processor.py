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
        drawing commands. Crops and saves each region separately.

        Uses text blocks as hard separators so two charts or tables
        sitting next to each other are never merged into one crop.
        """

        drawings = page.get_drawings()

        if not drawings:
            return

        page_rect = page.rect

        # Text block rects act as separators — any text between two
        # drawing clusters means they are distinct visual elements
        separator_rects = [block["rect"] for block in text_blocks]

        regions = _cluster_rects(
            [fitz.Rect(d["rect"]) for d in drawings],
            gap_threshold=20,
            page_rect=page_rect,
            separator_rects=separator_rects,
            max_cluster_width_ratio=0.6,
            max_cluster_height_ratio=0.6
        )

        for region_idx, region_rect in enumerate(regions):

            # Skip regions that are nearly the full page (background fills)
            region_area = region_rect.width * region_rect.height
            page_area = page_rect.width * page_rect.height

            if region_area > page_area * 0.85:
                continue

            # Skip tiny regions
            if region_area < MIN_IMAGE_AREA:
                continue

            # Expand slightly for padding, clamped to page bounds
            clip = region_rect + (-4, -4, 4, 4)
            clip = clip & page_rect

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

def _cluster_rects(rects, gap_threshold=20, page_rect=None,
                   separator_rects=None,
                   max_cluster_width_ratio=0.6,
                   max_cluster_height_ratio=0.6):
    """
    Groups drawing rects into clusters representing distinct visual elements.

    Key behaviours vs the naive single-linkage approach:
    -------------------------------------------------------
    1. COMPLETE-LINKAGE merge check — two groups only merge if their
       fully-merged bounding box is within gap_threshold of EVERY member
       of both groups. This prevents the chain-reaction bridging that causes
       two side-by-side charts to collapse into one big rect.

    2. TEXT SEPARATOR VETO — if a text block (heading, caption, label)
       falls between two candidate groups, they are not merged even if
       they would otherwise be close enough. Text between elements is a
       reliable signal of a boundary.

    3. SIZE CAP — if merging two groups would produce a rect wider than
       max_cluster_width_ratio * page_width OR taller than
       max_cluster_height_ratio * page_height, the merge is rejected.
       Prevents adjacent charts from becoming a single region.

    Parameters
    ----------
    rects               : list of fitz.Rect — drawing element bboxes
    gap_threshold       : max gap (pts) between elements in the same cluster
    page_rect           : fitz.Rect of the full page (used for size cap)
    separator_rects     : list of fitz.Rect for text blocks (separator veto)
    max_cluster_width_ratio  : max merged width as fraction of page width
    max_cluster_height_ratio : max merged height as fraction of page height
    """

    if not rects:
        return []

    separator_rects = separator_rects or []

    # Each cluster is a list of rects
    clusters = [[r] for r in rects]

    def bounding_rect(cluster):
        return fitz.Rect(
            min(r.x0 for r in cluster),
            min(r.y0 for r in cluster),
            max(r.x1 for r in cluster),
            max(r.y1 for r in cluster)
        )

    def gap_between(r1, r2):
        """Axis-aligned gap between two rects (0 if overlapping)."""
        dx = max(0.0, max(r1.x0, r2.x0) - min(r1.x1, r2.x1))
        dy = max(0.0, max(r1.y0, r2.y0) - min(r1.y1, r2.y1))
        return dx, dy

    def text_separates(br_a, br_b):
        """
        Returns True if any text block rect lies between the two
        bounding rects, i.e. it overlaps the gap region between them.
        """
        gap_x0 = min(br_a.x1, br_b.x1)
        gap_x1 = max(br_a.x0, br_b.x0)
        gap_y0 = min(br_a.y1, br_b.y1)
        gap_y1 = max(br_a.y0, br_b.y0)

        for sep in separator_rects:
            # Sep must fall within the gap region between the two clusters
            if (sep.x0 < gap_x1 and sep.x1 > gap_x0 and
                    sep.y0 < gap_y1 and sep.y1 > gap_y0):
                return True
        return False

    def can_merge(cluster_a, cluster_b):
        """
        Two clusters can merge only if:
          - Their bounding boxes are within gap_threshold of each other
          - No text block separates them
          - The merged bounding box doesn't exceed the size cap
        """
        br_a = bounding_rect(cluster_a)
        br_b = bounding_rect(cluster_b)

        dx, dy = gap_between(br_a, br_b)

        if dx > gap_threshold or dy > gap_threshold:
            return False

        if text_separates(br_a, br_b):
            return False

        if page_rect:
            merged = fitz.Rect(
                min(br_a.x0, br_b.x0), min(br_a.y0, br_b.y0),
                max(br_a.x1, br_b.x1), max(br_a.y1, br_b.y1)
            )
            if (merged.width > page_rect.width * max_cluster_width_ratio or
                    merged.height > page_rect.height * max_cluster_height_ratio):
                return False

        return True

    # Iteratively merge until no more merges are possible
    changed = True
    while changed:
        changed = False
        merged_clusters = []
        used = [False] * len(clusters)

        for i in range(len(clusters)):
            if used[i]:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if can_merge(current, clusters[j]):
                    current = current + clusters[j]
                    used[j] = True
                    changed = True
            merged_clusters.append(current)
            used[i] = True

        clusters = merged_clusters

    return [bounding_rect(c) for c in clusters]


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