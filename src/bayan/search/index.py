"""Lab 5: versioned FAISS index build for bilingual case retrieval."""

import json
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/search/bayan_cases.csv"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROC_VERSION = "bayan_search_v1"


def normalize_text(text: str) -> str:
    """Apply deterministic lightweight preprocessing for search."""
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.split()).strip()


def build_index(
    prefix: str,
    limit: int | None = None,
    model_name: str = DEFAULT_MODEL,
):
    """Encode cases, L2-normalise embeddings, and persist FAISS artifacts."""
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit).copy()

    if df.empty:
        raise ValueError("No cases available to index.")

    texts = [normalize_text(text) for text in df["case_text"].tolist()]

    encoder = SentenceTransformer(model_name)

    vectors = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype="float32")

    # Cosine similarity = inner product after L2 normalisation.
    faiss.normalize_L2(vectors)

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = Path(f"{prefix}_index.faiss")
    metadata_path = Path(f"{prefix}_metadata.jsonl")
    manifest_path = Path(f"{prefix}_manifest.json")

    faiss.write_index(index, str(index_path))

    metadata = df.to_dict(orient="records")
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": dim,
        "index_type": "IndexFlatIP",
        "normalised": True,
        "data_path": DATA_PATH,
        "index_file": index_path.name,
        "metadata_file": metadata_path.name,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest
