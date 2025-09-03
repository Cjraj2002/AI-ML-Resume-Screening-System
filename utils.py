import io
import re
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document

# ---- Clean extracted text ----
def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.lower()
    txt = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", txt)  # keep only readable chars
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ---- Extract text from PDF ----
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_chunks = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_chunks)

# ---- Extract text from DOCX ----
def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

# ---- Extract text from Image (OCR) ----
def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return pytesseract.image_to_string(image)

# ---- Smart extractor depending on file type ----
def extract_text(upload) -> str:
    name = (upload.name or "").lower()
    data = upload.read()
    upload.seek(0)

    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        return extract_text_from_docx(data)
    elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]):
        return extract_text_from_image(data)
    else:
        try:
            return extract_text_from_image(data)
        except Exception:
            return ""
