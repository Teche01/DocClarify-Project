from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from a PDF file.

    Returns:
        Extracted text and total number of pages.
    """
    pdf_reader = PdfReader(BytesIO(file_bytes))
    extracted_pages: list[str] = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                f"--- Page {page_number} ---\n{page_text.strip()}"
            )

    complete_text = "\n\n".join(extracted_pages)

    return complete_text, len(pdf_reader.pages)


def extract_docx_text(file_bytes: bytes) -> str:
    """Extract paragraph text from a DOCX file."""
    document = Document(BytesIO(file_bytes))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n\n".join(paragraphs)


def extract_txt_text(file_bytes: bytes) -> str:
    """Extract text from a TXT file."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


def extract_document_text(
    file_name: str,
    file_bytes: bytes,
) -> tuple[str, dict]:
    """
    Select the correct extraction method based on the file extension.
    """
    file_extension = Path(file_name).suffix.lower()

    if file_extension == ".pdf":
        text, page_count = extract_pdf_text(file_bytes)

        return text, {
            "file_type": "PDF",
            "page_count": page_count,
        }

    if file_extension == ".docx":
        text = extract_docx_text(file_bytes)

        return text, {
            "file_type": "DOCX",
            "page_count": None,
        }

    if file_extension == ".txt":
        text = extract_txt_text(file_bytes)

        return text, {
            "file_type": "TXT",
            "page_count": None,
        }

    raise ValueError(
        f"Unsupported file type: {file_extension}"
    )

def extract_pdf_pages(
    file_bytes: bytes,
) -> list[dict]:
    """
    Extract text separately from every PDF page.

    Returns:
        A list containing page number and page text.
    """

    pdf_reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page_number, page in enumerate(
        pdf_reader.pages,
        start=1,
    ):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(
                {
                    "page_number": page_number,
                    "text": page_text.strip(),
                }
            )

    return pages