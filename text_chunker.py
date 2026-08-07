from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split a long document into smaller overlapping text chunks.

    Args:
        text: Complete extracted document text.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Approximate repeated characters between chunks.

    Returns:
        A list containing the generated text chunks.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

def split_pdf_pages_into_chunks(
    pages: list[dict],
) -> list[dict]:
    """
    Split individual PDF pages into chunks while
    preserving their page numbers.
    """

    chunk_records = []

    global_chunk_index = 0

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = split_text_into_chunks(
            page_text
        )

        for chunk in page_chunks:
            chunk_records.append(
                {
                    "content": chunk,
                    "page_number": page_number,
                    "chunk_index": global_chunk_index,
                }
            )

            global_chunk_index += 1

    return chunk_records