import os
import base64
import fitz  # PyMuPDF


# --------------------------------------------------
# Thresholds
# --------------------------------------------------

MIN_IMAGE_AREA      = 5_000   # px² — skip tiny icons/bullets
CAPTION_PROXIMITY_PTS = 40    # pts — how close a caption must be to an image
LABEL_ABSORB_PTS    = 40      # pts — how far outside a cluster to absorb labels
MAX_LABEL_WORDS     = 12      # words — ignore paragraphs, keep axis labels


class ImageExtractor:
    """
    Extracts images from a single PDF page — both embedded XObjects
    (photos, logos) and vector/drawn regions (charts, diagrams).

    Each extracted region is registered in claimed_rects so the text
    extractor skips those areas.
    """

    def __init__(self, pdf_doc, pdf_path, base_dir, claimed_rects):
        self.doc           = pdf_doc
        self.pdf_path      = pdf_path
        self.base_dir      = base_dir
        self.claimed_rects = claimed_rects   # shared ref: page_num -> [fitz.Rect]
        self._seen_xrefs   = set()           # deduplicate shared XObjects

    # --------------------------------------------------
    # Public entry points
    # --------------------------------------------------

    def process_embedded(self, page, page_num, text_blocks):
        """
        Extract discrete image objects embedded in the PDF object tree.
        Deduplicates by xref so shared images (headers, logos) aren't
        repeated on every page.
        Returns list of item dicts.
        """
        items = []

        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]

            if xref in self._seen_xrefs:
                continue
            self._seen_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(self.doc, xref)
            except Exception as e:
                print(f"  ⚠ Could not read image xref={xref}: {e}")
                continue

            if pix.width * pix.height < MIN_IMAGE_AREA:
                continue

            file_name = os.path.join(
                self.base_dir, "images",
                f"{os.path.basename(self.pdf_path)}_xobj_{page_num}_{idx}_{xref}.png"
            )

            if not self._save_pixmap(pix, file_name):
                continue

            encoded = self._encode_image(file_name)

            image_rects = page.get_image_rects(xref)
            caption = ""
            if image_rects:
                img_rect  = image_rects[0]
                img_area  = img_rect.width * img_rect.height

                # Skip if this rect overlaps an already-claimed region
                if self._overlaps_claimed(img_rect, img_area,
                                          self.claimed_rects.get(page_num, [])):
                    continue

                caption = self._find_caption(img_rect, text_blocks)
                if page_num not in self.claimed_rects:
                    self.claimed_rects[page_num] = []
                self.claimed_rects[page_num].append(img_rect)

            items.append({
                "type": "image",
                "page": page_num,
                "caption": caption,
                "image": encoded,
                "path": file_name
            })

        return items

    def process_drawn(self, page, page_num, text_blocks):
        """
        Detect image-like regions painted with PDF drawing operators.
        These are invisible to get_images() but appear in drawing commands.
        Crops and saves each region separately.

        Uses text blocks as separators so two adjacent charts are never
        merged into one crop.
        Returns list of item dicts.
        """
        items = []
        drawings = page.get_drawings()

        if not drawings:
            return items

        page_rect = page.rect
        separator_rects = [block["rect"] for block in text_blocks]

        regions = cluster_rects(
            [fitz.Rect(d["rect"]) for d in drawings],
            gap_threshold=20,
            page_rect=page_rect,
            separator_rects=separator_rects,
            max_cluster_width_ratio=0.6,
            max_cluster_height_ratio=0.45
        )

        for region_idx, region_rect in enumerate(regions):

            region_area = region_rect.width * region_rect.height
            page_area   = page_rect.width * page_rect.height

            if region_area > page_area * 0.85:
                continue
            if region_area < MIN_IMAGE_AREA:
                continue

            # Re-read claimed_rects live so regions claimed earlier in this
            # loop are visible to later iterations
            if self._overlaps_claimed(region_rect, region_area,
                                      self.claimed_rects.get(page_num, [])):
                print(f"  ↷ Skipping drawn region {region_idx} page {page_num} — overlaps existing image")
                continue

            # Expand crop to absorb nearby axis labels / titles / footnotes
            clip = self._expand_crop(region_rect, regions, text_blocks, page_rect)

            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=clip)

            if pix.width * pix.height < MIN_IMAGE_AREA:
                continue

            file_name = os.path.join(
                self.base_dir, "images",
                f"{os.path.basename(self.pdf_path)}_region_{page_num}_{region_idx}.png"
            )

            if not self._save_pixmap(pix, file_name):
                continue

            encoded = self._encode_image(file_name)
            caption = self._find_caption(region_rect, text_blocks)

            if page_num not in self.claimed_rects:
                self.claimed_rects[page_num] = []
            self.claimed_rects[page_num].append(clip)

            items.append({
                "type": "image",
                "page": page_num,
                "caption": caption,
                "image": encoded,
                "path": file_name
            })

        return items

    def process_page_image(self, page, page_num):
        """
        Renders the full page as a high-res image.
        Fallback for scanned PDFs without a text layer.
        Returns list with one item dict, or empty list.
        """
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)

        file_name = os.path.join(
            self.base_dir, "page_images",
            f"page_{page_num:03d}.png"
        )

        if not self._save_pixmap(pix, file_name):
            return []

        encoded = self._encode_image(file_name)

        return [{
            "type": "page",
            "page": page_num,
            "caption": f"Full page snapshot — page {page_num + 1}",
            "image": encoded,
            "path": file_name
        }]

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _save_pixmap(self, pix, file_name):
        """Convert to RGB if needed, then save. Returns False if empty."""
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        pix.save(file_name)
        return os.path.getsize(file_name) > 0

    def _encode_image(self, file_name):
        with open(file_name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf8")

    def _find_caption(self, image_rect, text_blocks):
        """Find the closest text block directly above or below the image."""
        candidates = []

        for block in text_blocks:
            r = block["rect"]

            h_overlap = r.x0 < image_rect.x1 and r.x1 > image_rect.x0
            if not h_overlap:
                continue

            dist_above = image_rect.y0 - r.y1
            dist_below = r.y0 - image_rect.y1

            if 0 <= dist_above <= CAPTION_PROXIMITY_PTS:
                candidates.append((dist_above, block["text"]))
            elif 0 <= dist_below <= CAPTION_PROXIMITY_PTS:
                candidates.append((dist_below, block["text"]))

        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _overlaps_claimed(self, region_rect, region_area, claimed):
        """
        Returns True if region_rect significantly overlaps any claimed rect.
        Uses the smaller of the two areas as denominator (symmetric) so a
        large new region doesn't slip past a small claimed rect, and vice versa.
        """
        for cr in claimed:
            intersection = region_rect & cr
            if not intersection.is_empty:
                inter_area  = intersection.width * intersection.height
                cr_area     = cr.width * cr.height
                smaller     = min(region_area, cr_area) + 1e-6
                if inter_area / smaller > 0.4:
                    return True
        return False

    def _expand_crop(self, region_rect, all_regions, text_blocks, page_rect):
        """
        Expand the crop rect to absorb nearby axis labels, titles, and
        footnotes that sit just outside the drawn boundary.
        Only absorbs short text blocks (≤ MAX_LABEL_WORDS words) that
        won't bleed into an adjacent cluster.
        """
        expanded = fitz.Rect(region_rect)

        for block in text_blocks:
            br = block["rect"]

            if len(block["text"].split()) > MAX_LABEL_WORDS:
                continue

            dx = max(0.0, max(region_rect.x0, br.x0) - min(region_rect.x1, br.x1))
            dy = max(0.0, max(region_rect.y0, br.y0) - min(region_rect.y1, br.y1))

            if dx > LABEL_ABSORB_PTS or dy > LABEL_ABSORB_PTS:
                continue

            candidate = expanded | br

            overlaps_other = any(
                other != region_rect and
                candidate.intersects(other) and
                not region_rect.intersects(other)
                for other in all_regions
            )

            if not overlaps_other:
                expanded = candidate

        clip = expanded + (-6, -6, 6, 6)
        return clip & page_rect


# --------------------------------------------------
# Rect clustering (module-level — used by ImageExtractor)
# --------------------------------------------------

def cluster_rects(rects, gap_threshold=20, page_rect=None,
                  separator_rects=None,
                  max_cluster_width_ratio=0.6,
                  max_cluster_height_ratio=0.45):
    """
    Groups drawing rects into clusters representing distinct visual elements.

    Three merge guards:
      1. COMPLETE-LINKAGE — merged bbox must be within gap_threshold of
         every member of both clusters (prevents chain-reaction bridging).
      2. TEXT SEPARATOR VETO — any text block in the gap between two
         clusters blocks the merge.
      3. SIZE CAP — merged rect must not exceed the width/height ratio
         thresholds (prevents two stacked charts merging into one).
    """
    if not rects:
        return []

    separator_rects = separator_rects or []
    clusters = [[r] for r in rects]

    def bounding_rect(cluster):
        return fitz.Rect(
            min(r.x0 for r in cluster), min(r.y0 for r in cluster),
            max(r.x1 for r in cluster), max(r.y1 for r in cluster)
        )

    def gap_between(r1, r2):
        dx = max(0.0, max(r1.x0, r2.x0) - min(r1.x1, r2.x1))
        dy = max(0.0, max(r1.y0, r2.y0) - min(r1.y1, r2.y1))
        return dx, dy

    def text_separates(br_a, br_b):
        gap_x0 = min(br_a.x0, br_b.x0)
        gap_x1 = max(br_a.x1, br_b.x1)
        gap_y0 = min(br_a.y1, br_b.y1)
        gap_y1 = max(br_a.y0, br_b.y0)
        if gap_y1 <= gap_y0:
            return False
        return any(
            sep.x0 < gap_x1 and sep.x1 > gap_x0 and
            sep.y0 < gap_y1 and sep.y1 > gap_y0
            for sep in separator_rects
        )

    def can_merge(a, b):
        br_a, br_b = bounding_rect(a), bounding_rect(b)
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
            if (merged.width  > page_rect.width  * max_cluster_width_ratio or
                    merged.height > page_rect.height * max_cluster_height_ratio):
                return False
        return True

    changed = True
    while changed:
        changed = False
        merged, used = [], [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            cur = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if can_merge(cur, clusters[j]):
                    cur = cur + clusters[j]
                    used[j] = True
                    changed = True
            merged.append(cur)
            used[i] = True
        clusters = merged

    return [bounding_rect(c) for c in clusters]