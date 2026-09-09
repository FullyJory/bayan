"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.search.index import PREPROC_VERSION, normalize_text


DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        self.prefix = prefix

        manifest_path = Path(f"{prefix}_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing index manifest: {manifest_path}")

        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        required = {"model", "preproc_version", "n_vectors", "dim"}
        missing = required - self.manifest.keys()
        if missing:
            raise ValueError(
                f"Manifest missing required fields: {sorted(missing)}"
            )

        if self.manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError(
                "Search preprocessing version does not match index manifest."
            )

        index_path = Path(f"{prefix}_index.faiss")
        metadata_path = Path(f"{prefix}_metadata.jsonl")

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing index metadata: {metadata_path}"
            )

        self.index = faiss.read_index(str(index_path))

        with metadata_path.open(encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f if line.strip()]

        if self.index.ntotal != self.manifest["n_vectors"]:
            raise ValueError("FAISS vector count does not match manifest.")

        if self.index.d != self.manifest["dim"]:
            raise ValueError("FAISS dimension does not match manifest.")

        if len(self.metadata) != self.manifest["n_vectors"]:
            raise ValueError("Metadata count does not match manifest.")

        self.encoder = SentenceTransformer(self.manifest["model"])
        self.reranker = CrossEncoder(DEFAULT_RERANKER)

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        if not isinstance(query, str) or not query.strip():
            return []

        query = normalize_text(query)

        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        # Must match the L2-normalised index build path.
        faiss.normalize_L2(query_vector)

        candidate_count = min(candidates, self.index.ntotal)

        bi_scores, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        candidates_data = []

        for idx, bi_score in zip(indices[0], bi_scores[0]):
            if idx < 0:
                continue

            item = dict(self.metadata[int(idx)])
            item["bi_score"] = float(bi_score)
            candidates_data.append(item)

        if not candidates_data:
            return []

        pairs = [
            [query, item["case_text"]]
            for item in candidates_data
        ]

        rerank_scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        for item, score in zip(candidates_data, rerank_scores):
            item["score"] = float(score)

        candidates_data.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        results = [
            item
            for item in candidates_data
            if item["score"] >= min_score
        ]

        return results[:k]
