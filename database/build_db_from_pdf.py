# build_db_from_pdf.py  —  VietOCR version
# SETUP:
#   Download poppler current version is 25.12.0→ put in project_root/poppler/Library/bin/
#
# USAGE:
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python database/build_db_from_pdf.py

import os
import unicodedata, re
import shutil
import numpy as np
import cv2
import torch

from pypdf import PdfReader
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import ImageEnhance
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "luat-doanh-nghiep-2020_20_281_29.pdf")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
POPPLER_PATH = os.path.join(BASE_DIR, "poppler", "Library", "bin")
RAW_TXT_PATH = os.path.join(BASE_DIR, "ocr_raw_output.txt")
DEVICE_CFG = os.path.join(BASE_DIR, ".device_config")

DPI = 250  # increased from 200 → better OCR accuracy
BATCH_SIZE = 5  # pages per batch (reduce to 2 if RAM < 8GB)
INSERT_BATCH = 32  # chromadb insert batch

# ── Auto-detect GPU from install.bat config ──
# Reads .device_config written by install.bat
# Falls back to torch auto-detection if file missing
def detect_device():
    # Check .device_config written by install.bat
    if os.path.exists(DEVICE_CFG):
        with open(DEVICE_CFG, "r") as f:
            cfg_content = f.read().strip()
        # Parse DEVICE=xxx
        for line in cfg_content.splitlines():
            if line.startswith("DEVICE="):
                value = line.split("=", 1)[1].strip().lower()
                if value == "cuda":
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        print(f"  GPU detected: {gpu_name}")
                        return "cuda"
                    else:
                        print("  Warning: .device_config says cuda but no GPU found → using cpu")
                        return "cpu"
                else:
                    print("  Device config: cpu")
                    return "cpu"
    # Fallback: auto-detect via torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU auto-detected: {gpu_name}")
        return "cuda"
    print("  No GPU found → using cpu")
    return "cpu"

# =========================
# STEP 1: LOAD VIETOCR
# =========================
print("Loading VietOCR model (downloads ~300MB first time)...")
print("Detecting device...")

DEVICE = detect_device()
print(f"  Using device: {DEVICE.upper()}")
print()

config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = DEVICE

# ── Key accuracy improvements ──
# beamsearch=True → considers multiple candidate sequences, picks best one
# Much more accurate than greedy (False)
# GPU: ~15 min for 141 pages | CPU: ~2 hours
config['predictor']['beamsearch'] = True

detector = Predictor(config)
print(f"VietOCR loaded! (device={DEVICE.upper()}, beamsearch=True)\n")

# =========================
# STEP 2: IMAGE PREPROCESSING
# ─────────────────────────────
# Enhance image before OCR:
# 1. Convert to grayscale
# 2. Boost contrast  → makes text darker, background whiter
# 3. Boost sharpness → clearer character edges
# =========================
def preprocess_image(pil_image):
    # Convert to grayscale
    img = pil_image.convert('L')
    # Boost contrast (1.0=original, 2.0=double contrast)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    # Boost sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    # Convert back to RGB (VietOCR expects RGB)
    img = img.convert('RGB')
    return img

# =========================
# STEP 3: LINE DETECTION
# ─────────────────────────────
# Uses OpenCV to detect text line bounding boxes.
# Much more accurate than manual pixel projection because:
# - Handles indented text, numbered lists, italic lines
# - Merges characters into proper word/line groups
# - Skips decorative elements and noise automatically
# =========================
def detect_lines_opencv(pil_image):
    """
    Two-stage line detection tuned for Luật Doanh nghiệp 2020:

    Stage 1 — Single-line detection (kernel 80x1, iter=1):
      Finds individual text lines using tight horizontal dilation.
      Each line ~75-80px tall at 300 DPI, gap ~33px between lines.
      VietOCR is designed for single-line input — this gives best accuracy.

    Stage 2 — Tall box splitting:
      If a detected box is taller than MAX_LINE_HEIGHT, it means
      multiple lines merged. We split it by re-running line detection
      only within that region, preventing garbled multi-line OCR.

    Returns list of (y1, y2) row ranges sorted top to bottom.
    """
    try:

        img_array = np.array(pil_image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        # Otsu binarization — works well for scanned legal docs
        # Inverts so text = white (255), background = black (0)
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Stage 1: tight kernel → individual lines
        # width=80px connects chars/words in same line
        # height=1px keeps adjacent lines separate (gap ~33px >> 1px)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Max height for a single line at 300 DPI
        # avg=60px, max measured=92px → use 110px as ceiling
        MAX_LINE_H = 110

        # ── Page boundary filters ──────────────────────────────────
        # Measured from actual PDF pages (300 DPI, 2488x3492px):
        # - Binding artifacts (dots, smudges) always appear above y=300
        # - Real text always starts at y=300-320
        # - Left binding margin: x < 150px
        # - Right margin smudges: x+w > 2350px
        # - Real text lines are always wider than 150px
        # ───────────────────────────────────────────────────────────
        TOP_MARGIN   = 320   # skip page number + all binding artifacts
        LEFT_MARGIN  = 150   # skip left binding area
        RIGHT_EDGE   = img_w - 150  # skip right margin
        MIN_WIDTH    = 150   # real text is always wider than this

        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (h >= 10 and
                w >= MIN_WIDTH and
                x >= LEFT_MARGIN and
                (x + w) <= RIGHT_EDGE and
                y >= TOP_MARGIN):
                raw_boxes.append((y, y + h))

        raw_boxes.sort(key=lambda b: b[0])

        # Stage 2: split tall boxes, skip non-text regions
        # MAX_LINE_H: single text line ceiling (measured: avg=67px, max=92px)
        # SKIP_H: boxes taller than this are stamps/seals/signatures → skip
        MAX_LINE_H = 110   # single line ceiling
        SKIP_H     = 400   # stamp/seal/signature area → skip entirely

        final_boxes = []
        for y1, y2 in raw_boxes:
            box_h = y2 - y1

            # Skip stamp, seal, signature, table areas — too tall to be text
            if box_h > SKIP_H:
                continue

            if box_h <= MAX_LINE_H:
                # Normal single line — add padding for Vietnamese diacritics
                final_boxes.append((
                    max(0, y1 - 4),
                    min(img_h, y2 + 4)
                ))
            else:
                # Merged box (e.g. Điều title + artifact dots below it)
                # Re-detect individual lines within this region
                region = binary[y1:y2, :]
                sub_dilated = cv2.dilate(region, kernel, iterations=1)
                sub_cnts, _ = cv2.findContours(
                    sub_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                sub_boxes = []
                for sc in sub_cnts:
                    sx, sy, sw, sh = cv2.boundingRect(sc)
                    if sh >= 10 and sw >= MIN_WIDTH:
                        sub_boxes.append((y1 + sy, y1 + sy + sh))

                if sub_boxes:
                    sub_boxes.sort(key=lambda b: b[0])
                    for sy1, sy2 in sub_boxes:
                        sh = sy2 - sy1
                        # Skip sub-boxes that are still too tall (e.g. stamps)
                        if sh <= MAX_LINE_H:
                            final_boxes.append((
                                max(0, sy1 - 4),
                                min(img_h, sy2 + 4)
                            ))
                else:
                    # Fallback: keep if reasonable height
                    if box_h <= MAX_LINE_H * 2:
                        final_boxes.append((max(0, y1-4), min(img_h, y2+4)))

        final_boxes.sort(key=lambda b: b[0])
        return final_boxes

    except ImportError:
        return None

def detect_lines_projection(pil_image):
    """
    Fallback: horizontal pixel projection line detection.
    Less accurate than OpenCV but has no extra dependencies.
    """
    gray = pil_image.convert('L')
    arr  = np.array(gray)
    row_darkness = (arr < 200).sum(axis=1)

    in_line     = False
    line_starts = []
    line_ends   = []

    for i, dark in enumerate(row_darkness):
        if dark > 5 and not in_line:
            in_line = True
            line_starts.append(max(0, i - 4))
        elif dark <= 2 and in_line:
            in_line = False
            line_ends.append(min(arr.shape[0], i + 4))

    if in_line:
        line_ends.append(arr.shape[0])

    return list(zip(line_starts, line_ends))

# =========================
# STEP 4: OCR FUNCTION
# =========================
def ocr_page(pil_image):
    """
    1. Preprocess image (contrast + sharpness boost)
    2. Detect text line bounding boxes (OpenCV preferred, projection fallback)
    3. OCR each line with VietOCR
    4. Join lines into full page text
    """
    # Preprocess for better accuracy
    pil_image = preprocess_image(pil_image)
    width     = pil_image.width
    height    = pil_image.height

    # Try OpenCV line detection first (more accurate)
    boxes = detect_lines_opencv(pil_image)

    if not boxes:
        # Fallback to pixel projection
        boxes = detect_lines_projection(pil_image)

    # No lines detected → process whole page at once
    if not boxes:
        return detector.predict(pil_image)

    lines_text = []
    for y1, y2 in boxes:
        line_height = y2 - y1
        if line_height < 10:
            continue
        line_img = pil_image.crop((0, y1, width, y2))
        try:
            text = detector.predict(line_img)
            if text:
                text = text.strip()
                if is_valid_line(text):
                    lines_text.append(text)
        except Exception:
            pass

    return "\n".join(lines_text)


def is_valid_line(text: str) -> bool:
    """
    Filter out OCR garbage caused by scanning artifacts (binding dots,
    smudges, page creases). These produce lines like:
      Cons, COLIns, MIRComphis  → no Vietnamese vowel
      038000001, 0000000000     → mostly digits
      Đượng ở ..... ... . ...  → text + long whitespace + dots
    3 rules applied:
      1. Less than 30% real letters → garbage
      2. Short (1-2 words) with no Vietnamese vowel → binding smudge
      3. Text + 5+ spaces + dots/commas → garbled line detection
    """
    if not text or len(text) < 3:
        return False

    letter_count = sum(1 for c in text if unicodedata.category(c).startswith('L'))
    total = len(text)

    # Rule 1: mostly digits/symbols → number garbage
    if letter_count / total < 0.3:
        return False

    # Rule 2: short line with no Vietnamese vowel → binding smudge
    words = text.split()
    if len(words) <= 2:
        vowels = set('aăâeêiouôơưAĂÂEÊIOUÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ')
        if not any(c in vowels for c in text):
            return False

    # Rule 3: text + long whitespace + dots → garbled multi-line OCR
    if re.search(r'\s{5,}[.\s,]{5,}', text):
        return False

    return True

# =========================
# STEP 4: OCR ALL PAGES
# =========================
total_pages = len(PdfReader(PDF_PATH).pages)

est = "~15 min" if DEVICE == "cuda" else "~2 hours"
print(f"PDF      : {PDF_PATH}")
print(f"Pages    : {total_pages}")
print(f"DPI      : {DPI}  |  Batch: {BATCH_SIZE}  |  Device: {DEVICE.upper()}")
print(f"Estimated time: {est}")
print("Starting OCR...\n")

all_text_by_page = {}

for batch_start in range(1, total_pages + 1, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE - 1, total_pages)
    print(f"  Pages {batch_start:3d}–{batch_end:3d} / {total_pages} ...", end=" ", flush=True)

    pages = convert_from_path(
        PDF_PATH,
        dpi=DPI,
        first_page=batch_start,
        last_page=batch_end,
        fmt='jpeg',
        poppler_path=POPPLER_PATH
    )

    for i, page_img in enumerate(pages):
        page_num = batch_start + i
        text = ocr_page(page_img)
        all_text_by_page[page_num] = text.strip()
        print(".", end="", flush=True)

    del pages
    print(" done")

print(f"\nOCR complete — {len(all_text_by_page)} pages processed.")

# Save raw OCR text for inspection
with open(RAW_TXT_PATH, "w", encoding="utf-8") as f:
    for p in sorted(all_text_by_page):
        f.write(f"\n\n=== TRANG {p} ===\n")
        f.write(all_text_by_page[p])

print(f"Raw OCR saved → {RAW_TXT_PATH}")
print("  ↳ Open this file to check quality before continuing!\n")

# =========================
# STEP 5: JOIN ALL PAGES → SEGMENT BY ARTICLE
# ─────────────────────────────────────────────
# Join ALL pages first so articles spanning
# multiple pages are captured as one complete chunk
# =========================
print("Segmenting by legal article (Điều)...")

full_text = "\n".join(all_text_by_page[p] for p in sorted(all_text_by_page))

# Clean up whitespace
full_text = re.sub(r'\n{3,}', '\n\n', full_text)
full_text = re.sub(r'[ \t]+', ' ', full_text).strip()

# Split on "Điều X." — lookahead keeps delimiter at start of each segment
pattern    = r'(?:(?:^|\n)(?=Điều\s+\d+[a-z]?[.,]\s))'
raw_splits = re.split(pattern, full_text, flags=re.MULTILINE)

segments_raw = [s.strip() for s in raw_splits if len(s.strip()) > 50]
print(f"Found {len(segments_raw)} legal article segments.")

# Fallback: fixed-size chunks if article detection fails
if len(segments_raw) < 10:
    print("Warning: few articles detected — using fixed-size chunking as fallback.")
    chunk_size   = 3000
    overlap      = 300
    segments_raw = []
    i = 0
    while i < len(full_text):
        segments_raw.append(full_text[i:i + chunk_size])
        i += chunk_size - overlap
    print(f"  Created {len(segments_raw)} chunks.")

# =========================
# STEP 6: BUILD DOCUMENTS
# =========================
print("Building document objects...")

docs = []
for i, seg in enumerate(segments_raw):
    text = seg.strip()
    if len(text) < 30:
        continue

    article_match = re.match(r'Điều\s+(\d+[a-z]?)[\.\s]', text)
    article_num   = article_match.group(1) if article_match else str(i + 1)

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    title = lines[0][:120] if lines else f"Điều {article_num}"

    docs.append(Document(
        page_content=text,          # full text, no limit
        metadata={
            "so_ky_hieu":     "59/2020/QH14",
            "loai_van_ban":   "Luật",
            "title":          title,
            "article_number": article_num,
            "nguon_thu_thap": "Luật Doanh nghiệp 2020",
            "char_count":     len(text),
            "segment_index":  i,
        }
    ))

print(f"Built {len(docs)} document segments.")
if docs:
    avg_len = sum(len(d.page_content) for d in docs) // len(docs)
    max_len = max(len(d.page_content) for d in docs)
    min_len = min(len(d.page_content) for d in docs)
    print(f"  Average chars : {avg_len:,}")
    print(f"  Longest       : {max_len:,} chars")
    print(f"  Shortest      : {min_len:,} chars")

# =========================
# STEP 7: EMBED + CHROMADB
# =========================
print("\nLoading embedding model (BAAI/bge-small-en-v1.5)...")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print(f"Building ChromaDB at: {DB_PATH}")

if os.path.exists(DB_PATH):
    print("  Clearing old ChromaDB...")
    shutil.rmtree(DB_PATH)

vs = None
for i in range(0, len(docs), INSERT_BATCH):
    chunk = docs[i:i + INSERT_BATCH]
    if vs is None:
        vs = Chroma.from_documents(chunk, embedding, persist_directory=DB_PATH)
    else:
        vs.add_documents(chunk)
    print(f"  Indexed {min(i + INSERT_BATCH, len(docs))}/{len(docs)} segments")

print("\n✅ DONE!")
print(f"   Articles indexed  : {len(docs)}")
print(f"   ChromaDB path     : {DB_PATH}")
print(f"   Raw OCR text      : {RAW_TXT_PATH}")
print("\nNext step: python database/build_db_from_txt.py  (to rebuild DB anytime)")
print("Or run  : python app.py")