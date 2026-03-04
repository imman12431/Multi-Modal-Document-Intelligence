import os
import base64
import fitz  # PyMuPDF
import tabula
import re


MIN_ROWS     = 2
MIN_COLS     = 2
MIN_AREA_PTS = 50   # minimum width AND height in pts


class TableExtractor:
    """
    Detects tables using tabula, then saves each detected region as a
    high-resolution image for the vision model to read.
    """

    def __init__(self, pdf_path, base_dir, claimed_rects):
        self.pdf_path      = pdf_path
        self.base_dir      = base_dir
        self.claimed_rects = claimed_rects

    # --------------------------------------------------
    # Public entry point
    # --------------------------------------------------

    def process_page(self, page_num):
        items = []

        try:
            raw = tabula.read_pdf(
                self.pdf_path,
                pages=page_num + 1,
                multiple_tables=True,
                silent=True,
                output_format="json"
            )
        except Exception as e:
            print(f"  ⚠ Tabula failed page {page_num}: {e}")
            return items

        if not raw:
            return items

        doc  = fitz.open(self.pdf_path)
        page = doc[page_num]

        for idx, table_data in enumerate(raw):
            try:
                items += self._extract_one(page, page_num, idx, table_data)
            except Exception as e:
                print(f"  ⚠ Table crop failed page {page_num} idx {idx}: {e}")

        doc.close()
        return items

    # --------------------------------------------------
    # Validate — reject false positives
    # --------------------------------------------------

    def _is_real_table(self, data, rect):
        if not data or len(data) < MIN_ROWS:
            return False

        max_cols = max(len(row) for row in data)
        if max_cols < MIN_COLS:
            return False

        if rect.width < MIN_AREA_PTS or rect.height < MIN_AREA_PTS:
            return False

        # Must contain at least one numeric cell
        num_pat = re.compile(r'-?\d+\.?\d*')
        has_number = any(
            num_pat.search(c.get("text", ""))
            for row in data for c in row
        )
        if not has_number:
            return False

        return True

    # --------------------------------------------------
    # Crop and save one table as an image
    # --------------------------------------------------

    def _extract_one(self, page, page_num, idx, table_data):
        data = table_data.get("data", [])
        if not data:
            return []

        all_tops    = [c["top"]                for row in data for c in row]
        all_lefts   = [c["left"]               for row in data for c in row]
        all_bottoms = [c["top"]  + c["height"] for row in data for c in row]
        all_rights  = [c["left"] + c["width"]  for row in data for c in row]

        top, left     = min(all_tops),    min(all_lefts)
        bottom, right = max(all_bottoms), max(all_rights)

        new_rect = fitz.Rect(left, top, right, bottom)

        if not self._is_real_table(data, new_rect):
            print(f"  ↷ Skipping non-table page {page_num} idx {idx}")
            return []

        new_area = new_rect.width * new_rect.height

        # Skip duplicates
        for cr in self.claimed_rects.get(page_num, []):
            inter = new_rect & cr
            if not inter.is_empty:
                overlap = (inter.width * inter.height) / (new_area + 1e-6)
                if overlap > 0.5:
                    print(f"  ↷ Skipping duplicate table page {page_num} idx {idx}")
                    return []

        if page_num not in self.claimed_rects:
            self.claimed_rects[page_num] = []
        self.claimed_rects[page_num].append(new_rect)

        pad  = 6
        clip = fitz.Rect(left - pad, top - pad, right + pad, bottom + pad)
        clip = clip & page.rect

        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=clip)

        if pix.n > 4 or pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        file_name = os.path.join(
            self.base_dir, "tables",
            f"{os.path.basename(self.pdf_path)}_table_{page_num}_{idx}.png"
        )
        pix.save(file_name)

        if os.path.getsize(file_name) == 0:
            return []

        with open(file_name, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf8")

        print(f"  ✅ Table image saved — page {page_num} idx {idx} "
              f"({clip.width:.0f}×{clip.height:.0f} pts)")

        return [{
            "type":  "table",
            "page":  page_num,
            "text":  "",
            "image": encoded,
            "path":  file_name
        }]