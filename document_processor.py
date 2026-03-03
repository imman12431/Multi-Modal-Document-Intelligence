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
        self._claimed_rects = {}         # page_num → list of fitz.Rect already
                                         # captured as table/image — used to
                                         # suppress duplicate text extraction

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
    # TEXT — chunk and save, skipping claimed regions
    # --------------------------------------------------

    def _process_text(self, page, page_num):
        """
        Extracts text from the page, but skips any text block whose
        bounding rect significantly overlaps a region already claimed
        by a table or image extractor. This prevents axis labels,
        table cell text, and figure annotations from also appearing
        as standalone text chunks.
        """

        claimed = self._claimed_rects.get(page_num, [])

        # Collect only text blocks that are NOT inside a claimed region
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        clean_parts = []

        for block in raw.get("blocks", []):

            if block.get("type") != 0:   # skip image blocks
                continue

            block_rect = fitz.Rect(block["bbox"])

            # Check overlap with every claimed rect on this page
            overlaps = False
            for cr in claimed:
                intersection = block_rect & cr   # fitz intersection
                if not intersection.is_empty:
                    # Overlap ratio relative to the text block's own area
                    overlap_ratio = (intersection.width * intersection.height) / \
                                    (block_rect.width * block_rect.height + 1e-6)
                    if overlap_ratio > 0.3:   # >30% overlap → belongs to that region
                        overlaps = True
                        break

            if overlaps:
                continue

            # Collect text from this block
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")

            block_text = block_text.strip()
            if block_text:
                clean_parts.append(block_text)

        if not clean_parts:
            return

        full_text = "\n".join(clean_parts)
        chunks = self.text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):

            chunk = chunk.strip()
            if not chunk:
                continue

            file_name = os.path.join(
                self.base_dir, "text",
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
    # Format raw tabula output as a markdown table
    # --------------------------------------------------

    # --------------------------------------------------
    # Format raw tabula output as a markdown table
    # --------------------------------------------------

    def _format_table_as_markdown(self, raw_text):
        """
        Converts raw tabula pipe/CSV output to a markdown table.
        Stored at extraction time so both the summarizer and the
        QA system receive clean, structured input.
        """

        if not raw_text or not raw_text.strip():
            return raw_text

        lines = [l for l in raw_text.strip().split("\n") if l.strip()]

        if len(lines) < 2:
            return raw_text

        # Detect delimiter — tabula uses | (our config) or , by default
        delimiter = "|" if "|" in lines[0] else ","

        rows = [line.split(delimiter) for line in lines]

        # Normalize all rows to the same column count
        max_cols = max(len(row) for row in rows)
        rows = [row + [""] * (max_cols - len(row)) for row in rows]

        # Clean up whitespace in each cell
        rows = [[cell.strip() for cell in row] for row in rows]

        # Build markdown table
        header  = "| " + " | ".join(rows[0]) + " |"
        divider = "| " + " | ".join(["---"] * max_cols) + " |"
        body    = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])

        return f"{header}\n{divider}\n{body}"

    # --------------------------------------------------
    # Format 2D row list → markdown (pdfplumber output)
    # --------------------------------------------------

    def _format_table_as_markdown_rows(self, rows):
        """
        Converts pdfplumber's native 2D list directly to markdown.
        Avoids CSV round-trip so pipes/commas in cell text are preserved.
        """
        if not rows or len(rows) < 2:
            return ""

        max_cols = max(len(row) for row in rows)
        rows = [row + [""] * (max_cols - len(row)) for row in rows]
        rows = [[str(c).strip() if c is not None else "" for c in row]
                for row in rows]

        header  = "| " + " | ".join(rows[0]) + " |"
        divider = "| " + " | ".join(["---"] * max_cols) + " |"
        body    = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])

        return f"{header}\n{divider}\n{body}"

    # --------------------------------------------------
    # TABLES — pdfplumber primary, tabula fallback
    # --------------------------------------------------

    def _process_tables_tabula(self, page_num):
        """
        Two-stage table extraction:

        Stage 1 — pdfplumber with explicit column detection
            For each table region, pdfplumber snaps column boundaries
            from the x-positions of words in the header row. This handles
            borderless tables (like the Qatar macro indicators table) where
            tabula sees the whole row as one cell because there are no
            vertical ruling lines to split on.

        Stage 2 — tabula fallback
            Only runs if pdfplumber finds no tables at all on the page.
            Kept as a safety net for edge cases.

        Both stages claim bounding boxes into _claimed_rects so
        _process_text skips those regions.
        """

        import pdfplumber

        pdfplumber_saved = False

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                pl_page = pdf.pages[page_num]

                # Use explicit_vertical_lines strategy so pdfplumber can
                # handle borderless tables — derive column positions from
                # the x-coordinates of all words on the page
                words = pl_page.extract_words()

                if not words:
                    raise ValueError("no words on page")

                # Collect unique x0 positions of words — these are the
                # natural column start positions in the PDF
                x_positions = sorted(set(round(w["x0"]) for w in words))

                # Use pdfplumber's text-based table detection first
                tables = pl_page.find_tables()

                if not tables:
                    # Fall back to explicit vertical lines strategy
                    table_settings = {
                        "vertical_strategy": "explicit",
                        "horizontal_strategy": "lines",
                        "explicit_vertical_lines": x_positions,
                        "snap_tolerance": 5,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                    }
                    tables = pl_page.find_tables(table_settings)

                if not tables:
                    raise ValueError("pdfplumber found no tables")

                for idx, table_obj in enumerate(tables):

                    # Claim bbox
                    bbox = table_obj.bbox
                    if page_num not in self._claimed_rects:
                        self._claimed_rects[page_num] = []
                    self._claimed_rects[page_num].append(
                        fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                    )

                    rows = table_obj.extract()
                    if not rows or len(rows) < 2:
                        continue

                    # Replace None with empty string
                    rows = [[c if c is not None else "" for c in row]
                            for row in rows]

                    # Skip if it looks like the whole table collapsed
                    # into one column (tabula-style failure)
                    if len(rows[0]) < 3:
                        print(f"  [pdfplumber] page {page_num} table {idx}: "
                              f"skipped — only {len(rows[0])} col(s) detected")
                        continue

                    table_text = self._format_table_as_markdown_rows(rows)
                    if not table_text:
                        continue

                    file_name = os.path.join(
                        self.base_dir, "tables",
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

                    pdfplumber_saved = True
                    print(f"  ✅ [pdfplumber] Table saved — page {page_num}, "
                          f"{len(rows)-1} rows × {len(rows[0])} cols")

        except Exception as e:
            if "no tables" not in str(e) and "no words" not in str(e):
                print(f"  ⚠ pdfplumber failed page {page_num}: {e}")

        # ── Tabula fallback ──────────────────────────────────────────────
        if pdfplumber_saved:
            return

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

                table = table.fillna("")

                # Skip if tabula collapsed everything into one column
                if len(table.columns) < 3:
                    print(f"  [tabula] page {page_num} table {idx}: "
                          f"skipped — only {len(table.columns)} col(s)")
                    continue

                table_text = self._format_table_as_markdown(
                    table.to_csv(index=False, sep="|").strip()
                )

                if not table_text:
                    continue

                file_name = os.path.join(
                    self.base_dir, "tables",
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

                print(f"  ✅ [tabula] Table saved — page {page_num}, "
                      f"{len(table)} rows × {len(table.columns)} cols")

        except Exception as e:
            print(f"  ⚠ Tabula fallback failed page {page_num}: {e}")


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
                # Claim the image rect so text extraction and drawn region
                # detection both skip this area
                if page_num not in self._claimed_rects:
                    self._claimed_rects[page_num] = []
                self._claimed_rects[page_num].append(image_rects[0])

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
            max_cluster_height_ratio=0.45
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

            # Skip if this region significantly overlaps an already-claimed
            # embedded image XObject — prevents double-extracting a chart
            # that contains an embedded raster image inside its drawing frame
            already_claimed = self._claimed_rects.get(page_num, [])
            duplicate = False
            for cr in already_claimed:
                intersection = region_rect & cr
                if not intersection.is_empty:
                    overlap = (intersection.width * intersection.height) / \
                              (region_area + 1e-6)
                    if overlap > 0.5:
                        duplicate = True
                        break
            if duplicate:
                print(f"  ↷ Skipping drawn region {region_idx} on page {page_num} — overlaps existing image")
                continue

            # --------------------------------------------------
            # Expand the crop to include nearby text elements
            # (axis labels, tick values, titles, legends, footnotes)
            # that sit just outside the drawn boundary of the cluster.
            #
            # Strategy: absorb any text block whose rect is within
            # LABEL_ABSORB_PTS of the cluster bounding box, as long
            # as expanding to include it doesn't cause the crop to
            # overlap a different cluster.
            # --------------------------------------------------

            LABEL_ABSORB_PTS = 40   # how far outside the cluster to look
            MAX_LABEL_WORDS  = 12   # axis labels/titles are short — ignore paragraphs

            expanded = fitz.Rect(region_rect)  # start from cluster boundary

            for block in text_blocks:
                br = block["rect"]

                # Ignore long text blocks — body paragraphs, not annotations
                word_count = len(block["text"].split())
                if word_count > MAX_LABEL_WORDS:
                    continue

                # Is this text block close to the cluster?
                dx = max(0.0, max(region_rect.x0, br.x0) - min(region_rect.x1, br.x1))
                dy = max(0.0, max(region_rect.y0, br.y0) - min(region_rect.y1, br.y1))

                if dx > LABEL_ABSORB_PTS or dy > LABEL_ABSORB_PTS:
                    continue  # too far away

                # Would absorbing this block cause us to overlap another cluster?
                candidate = expanded | br   # union

                overlaps_other = any(
                    other_rect != region_rect and
                    candidate.intersects(other_rect) and
                    not region_rect.intersects(other_rect)
                    for other_rect in regions
                )

                if overlaps_other:
                    continue  # skip — would bleed into an adjacent figure

                expanded = candidate

            # Add a small fixed padding on top of the expanded rect,
            # then clamp to page bounds
            clip = expanded + (-6, -6, 6, 6)
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

            # Claim the expanded crop rect so text extraction skips this area
            if page_num not in self._claimed_rects:
                self._claimed_rects[page_num] = []
            self._claimed_rects[page_num].append(clip)

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

            # --- Tables first — claims their rects before text runs ---
            self._process_tables_tabula(page_num)

            # --- Embedded images — claims their rects ---
            self._process_embedded_images(page, page_num, text_blocks)

            # --- Drawn/vector regions — skips anything already claimed ---
            self._process_drawn_regions(page, page_num, text_blocks)

            # --- Text last — skips all claimed regions ---
            self._process_text(page, page_num)

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
                   max_cluster_height_ratio=0.45):
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
                               (0.45 means a single figure can be at most
                               45% of page height — prevents two stacked
                               half-page charts from merging)
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
        Returns True if any text block lies in the gap region between
        the two clusters.

        For two vertically stacked figures:

            ┌─────────────┐
            │  cluster A  │  y0=100  y1=300
            └─────────────┘
               "Figure 1"    ← caption at y0=310  y1=325   (in the gap)
            ┌─────────────┐
            │  cluster B  │  y0=340  y1=540
            └─────────────┘

        The vertical gap is y1 of the upper cluster → y0 of the lower:
            gap_y0 = min(br_a.y1, br_b.y1) = 300   (bottom of upper)
            gap_y1 = max(br_a.y0, br_b.y0) = 340   (top of lower)

        The horizontal span of the gap matches the shared width of the
        two clusters so a caption that only spans one column is still caught:
            gap_x0 = min(br_a.x0, br_b.x0)          (leftmost left edge)
            gap_x1 = max(br_a.x1, br_b.x1)          (rightmost right edge)

        A separator text block is inside the gap if it overlaps both axes.
        """

        # Horizontal span — union of both cluster widths
        gap_x0 = min(br_a.x0, br_b.x0)   # leftmost left edge
        gap_x1 = max(br_a.x1, br_b.x1)   # rightmost right edge

        # Vertical span — the space *between* the two clusters
        gap_y0 = min(br_a.y1, br_b.y1)   # bottom of the higher cluster
        gap_y1 = max(br_a.y0, br_b.y0)   # top of the lower cluster

        # If gap_y1 <= gap_y0 the clusters overlap vertically — no gap exists
        if gap_y1 <= gap_y0:
            return False

        for sep in separator_rects:
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