from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the local embedding model.

    Caching prevents the model from being loaded repeatedly
    while the application is running.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text chunks into numerical vectors.

    Args:
        texts: Document chunks that need to be embedded.

    Returns:
        A NumPy array containing one embedding per chunk.
    """
    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return np.empty(
            shape=(0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    model = load_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings