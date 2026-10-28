"""Turn uploaded files into clean text chunks (with page numbers for PDFs)."""
import io
import re
from dataclasses import dataclass

from . import config

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Chunk:
    text: str
    page: int | None
    heading: str | None


def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return [(page_number_or_None, text), ...]."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Use PDF, TXT or MD.")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(None, data.decode("utf-8", errors="replace"))]


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"-\n(\w)", r"\1", text)          # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, page: int | None = None,
               size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Paragraph-aware chunking. Chunks never cross a markdown heading and
    remember it, so answers can cite "Fees > Refund policy"."""
    text = _clean(text)
    chunks: list[Chunk] = []
    heading: str | None = None
    buf = ""
    fresh = True  # buf holds only overlap from the previous chunk

    def emit(keep_overlap: bool):
        nonlocal buf, fresh
        if buf.strip() and not fresh:
            chunks.append(Chunk(buf.strip(), page, heading))
        if keep_overlap and len(buf) > overlap > 0:
            tail = buf[-overlap:]
            buf = tail[tail.find(" ") + 1:] if " " in tail else tail
        else:
            buf = ""
        fresh = True

    def add(piece: str):
        nonlocal buf, fresh
        buf = f"{buf}\n\n{piece}" if buf else piece
        fresh = False

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"^#{1,6}\s+([^\n]*)", para)
        if m:
            emit(keep_overlap=False)
            heading = m.group(1).strip()
            para = para[m.end():].strip()
            if not para:
                continue
        while len(para) > size:  # very long paragraph: split at a sentence end
            cut = para.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if buf and not fresh:
                emit(keep_overlap=True)
            add(para[:cut].strip())
            emit(keep_overlap=True)
            para = para[cut:].strip()
        if not para:
            continue
        if buf and not fresh and len(buf) + len(para) + 2 > size:
            emit(keep_overlap=True)
        add(para)
    emit(keep_overlap=False)
    return chunks


def file_to_chunks(filename: str, data: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page, text in extract_pages(filename, data):
        chunks.extend(chunk_text(text, page))
    if not chunks:
        raise ValueError("No readable text found (scanned PDFs need OCR first).")
    return chunks
