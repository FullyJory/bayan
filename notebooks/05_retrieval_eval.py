"""Lab 5: labelled-query bilingual retrieval evaluation."""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayan.search.index import normalize_text

PREFIX = "artifacts/case_index_v1"
QUERIES_PATH = "data/search/bayan_queries.jsonl"
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def recall_at_10(retrieved, relevant):
    return float(bool(set(retrieved[:10]) & set(relevant)))


def reciprocal_rank_at_10(retrieved, relevant):
    relevant = set(relevant)
    for rank, case_id in enumerate(retrieved[:10], start=1):
        if case_id in relevant:
            return 1.0 / rank
    return 0.0


def main():
    manifest = json.loads(
        Path(f"{PREFIX}_manifest.json").read_text(encoding="utf-8")
    )

    index = faiss.read_index(f"{PREFIX}_index.faiss")

    with open(f"{PREFIX}_metadata.jsonl", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f if line.strip()]

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    encoder = SentenceTransformer(manifest["model"])
    reranker = CrossEncoder(RERANKER)

    answerable = [q for q in queries if not q["no_answer"]]
    no_answer = [q for q in queries if q["no_answer"]]

    rows = []
    bi_latencies = []
    rerank_latencies = []

    # ----- Answerable retrieval evaluation -----
    for q in answerable:
        query = normalize_text(q["query"])

        t0 = time.perf_counter()

        vector = encoder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype="float32")
        faiss.normalize_L2(vector)

        _, indices = index.search(vector, 50)

        bi_latencies.append((time.perf_counter() - t0) * 1000)

        candidate_ids = [
            metadata[int(i)]["case_id"]
            for i in indices[0]
            if i >= 0
        ]

        bi_top10 = candidate_ids[:10]

        t1 = time.perf_counter()

        pairs = [
            [query, metadata[int(i)]["case_text"]]
            for i in indices[0]
            if i >= 0
        ]

        scores = np.asarray(
            reranker.predict(pairs, show_progress_bar=False)
        )

        rerank_latencies.append((time.perf_counter() - t1) * 1000)

        order = np.argsort(scores)[::-1]
        reranked_ids = [candidate_ids[i] for i in order][:10]

        relevant = q["relevant_case_ids"]

        rows.append(
            {
                "lang": q["lang"],
                "bi_recall": recall_at_10(bi_top10, relevant),
                "bi_mrr": reciprocal_rank_at_10(bi_top10, relevant),
                "rr_recall": recall_at_10(reranked_ids, relevant),
                "rr_mrr": reciprocal_rank_at_10(reranked_ids, relevant),
            }
        )

    def mean(key, subset=None):
        selected = rows if subset is None else [
            r for r in rows if r["lang"] == subset
        ]
        return float(np.mean([r[key] for r in selected]))

    print("=== Retrieval Evaluation ===")
    print(f"Answerable queries: {len(answerable)}")
    print(f"Recall@10 without reranking: {mean('bi_recall'):.4f}")
    print(f"MRR@10 without reranking: {mean('bi_mrr'):.4f}")
    print(f"Recall@10 with reranking: {mean('rr_recall'):.4f}")
    print(f"MRR@10 with reranking: {mean('rr_mrr'):.4f}")

    ar_rr = mean("rr_mrr", "ar")
    en_rr = mean("rr_mrr", "en")
    gap = abs(ar_rr - en_rr)

    print()
    print("=== Language Slices (reranked MRR@10) ===")
    print(f"Arabic: {ar_rr:.4f}")
    print(f"English: {en_rr:.4f}")
    print(f"Cross-lingual slice gap: {gap:.4f}")

    print()
    print("=== Stage Latency ===")
    print(f"Bi-encoder p50 latency: {np.median(bi_latencies):.2f} ms/query")
    print(f"Cross-encoder p50 latency: {np.median(rerank_latencies):.2f} ms/query")

    # ----- No-answer threshold evaluation -----
    top_scores = []

    for q in no_answer:
        query = normalize_text(q["query"])

        vector = encoder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype="float32")
        faiss.normalize_L2(vector)

        _, indices = index.search(vector, 50)

        pairs = [
            [query, metadata[int(i)]["case_text"]]
            for i in indices[0]
            if i >= 0
        ]

        scores = np.asarray(
            reranker.predict(pairs, show_progress_bar=False)
        )

        top_scores.append(float(np.max(scores)))

    print()
    print("=== No-answer Threshold Behaviour ===")
    print(f"No-answer queries: {len(no_answer)}")

    thresholds = [-2.0, -1.0, 0.0, 0.25, 1.0, 2.0, 3.0, 4.0, 5.0]

    for threshold in thresholds:
        correct = sum(score < threshold for score in top_scores)
        print(
            f"min_score={threshold:>5.2f}: "
            f"{correct}/{len(no_answer)} correct empty results"
        )

    print()
    print(
        "No-answer top-score range: "
        f"{min(top_scores):.4f} to {max(top_scores):.4f}"
    )


if __name__ == "__main__":
    main()
