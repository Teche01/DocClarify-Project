from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any
from embedding_generator import generate_embeddings

import chromadb
import numpy as np

VECTOR_DATABASE_PATH = Path("vector_database")
COLLECTION_NAME = "rag_documents"


def create_document_id(file_bytes: bytes) -> str:
    """
    Create a stable unique ID using the document content.

    Uploading the same file again creates the same document ID,
    which helps prevent duplicate records.
    """
    return sha256(file_bytes).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_chroma_collection() -> Any:
    """
    Create or load the persistent ChromaDB collection.
    """
    database_path = VECTOR_DATABASE_PATH.resolve()

    database_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(database_path)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def store_document_chunks(
    document_id: str,
    file_name: str,
    file_type: str,
    page_count: int | None,
    chunks: list[str],
    embeddings: np.ndarray,
    page_numbers: list[int | None] | None = None,
) -> int:
    """
    Store document chunks, embeddings and metadata in ChromaDB.

    Returns:
        Number of chunks stored.
    """

    if not chunks:
        return 0

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "The number of chunks and embeddings must match."
        )

    if page_numbers is None:
        page_numbers = [
            None
            for _ in chunks
        ]

    if len(page_numbers) != len(chunks):
        raise ValueError(
            "The number of page numbers and chunks must match."
        )

    collection = get_chroma_collection()

    chunk_ids = [
        f"{document_id}_chunk_{chunk_index}"
        for chunk_index in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "source_file": file_name,
            "file_type": file_type,
            "page_count": int(page_count or 0),
            "page_number": int(
            page_numbers[chunk_index] or 0
            ),
            "chunk_index": chunk_index,
        }
        for chunk_index in range(len(chunks))
    ]

    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return len(chunk_ids)


def get_stored_chunk_count() -> int:
    """Return the total number of chunks stored in ChromaDB."""
    collection = get_chroma_collection()

    return collection.count()

def retrieve_relevant_chunks(
    question: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant document chunks for a question.

    Args:
        question: Question entered by the user.
        top_k: Maximum number of chunks to retrieve.

    Returns:
        A list containing retrieved chunks and their metadata.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        return []

    if top_k <= 0:
        raise ValueError(
            "The number of results must be greater than zero."
        )

    collection = get_chroma_collection()
    stored_count = collection.count()

    if stored_count == 0:
        return []

    # Use the same local embedding model that was used
    # for the document chunks.
    question_embeddings = generate_embeddings(
        [cleaned_question]
    )

    if question_embeddings.size == 0:
        return []

    number_of_results = min(top_k, stored_count)

    query_results = collection.query(
        query_embeddings=[
            question_embeddings[0].tolist()
        ],
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    document_groups = query_results.get("documents") or []
    metadata_groups = query_results.get("metadatas") or []
    distance_groups = query_results.get("distances") or []

    documents = (
        document_groups[0]
        if document_groups
        else []
    )

    metadatas = (
        metadata_groups[0]
        if metadata_groups
        else []
    )

    distances = (
        distance_groups[0]
        if distance_groups
        else []
    )

    retrieved_chunks: list[dict[str, Any]] = []

    for result_index, document_text in enumerate(documents):
        metadata = (
            metadatas[result_index]
            if result_index < len(metadatas)
            else {}
        )

        distance = (
            float(distances[result_index])
            if result_index < len(distances)
            else 0.0
        )

        retrieved_chunks.append(
            {
                "rank": result_index + 1,
                "content": document_text,
                "distance": distance,
                "document_id": metadata.get(
                    "document_id",
                    "",
                ),
                "source_file": metadata.get(
                    "source_file",
                    "Unknown document",
                ),
                "file_type": metadata.get(
                    "file_type",
                    "Unknown",
                ),
                "page_number": metadata.get(
                    "page_number",
                    0,
                ),
                "chunk_index": metadata.get(
                    "chunk_index",
                    0,
                ),
            }
        )

    return retrieved_chunks

def get_stored_documents() -> list[dict[str, Any]]:
    """
    Return a summary of documents currently stored in ChromaDB.
    """

    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    results = collection.get(
        include=["metadatas"]
    )

    metadatas = results.get("metadatas") or []

    documents: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:
        if not metadata:
            continue

        document_id = metadata.get(
            "document_id",
            "unknown"
        )

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "source_file": metadata.get(
                    "source_file",
                    "Unknown document"
                ),
                "file_type": metadata.get(
                    "file_type",
                    "Unknown"
                ),
                "chunk_count": 0,
            }

        documents[document_id]["chunk_count"] += 1

    return list(documents.values())


def clear_vector_database() -> None:
    """
    Delete all document chunks from the ChromaDB collection.
    """

    collection = get_chroma_collection()

    if collection.count() == 0:
        return

    stored_records = collection.get()

    stored_ids = stored_records.get("ids") or []

    if stored_ids:
        collection.delete(ids=stored_ids)

def delete_document(
    document_id: str,
) -> None:
    """
    Delete all chunks belonging to one document.
    """

    collection = get_chroma_collection()

    if not document_id:
        return

    collection.delete(
        where={
            "document_id": document_id
        }
    )

def is_document_stored(
    document_id: str,
) -> bool:
    """
    Check whether a document is already stored in ChromaDB.
    """

    collection = get_chroma_collection()

    results = collection.get(
        where={
            "document_id": document_id
        },
        limit=1,
    )

    stored_ids = results.get("ids") or []

    return len(stored_ids) > 0