# Model Card — Bayan Two-Stage Retrieval

## Intended use
Retrieve similar bilingual municipal cases using multilingual dense retrieval followed by cross-encoder reranking.

## Artefact / data versions
- Model/checkpoint: paraphrase-multilingual-MiniLM-L12-v2 + cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Preprocessing version: bayan_search_v1
- Data version/snapshot: 20,000-case Bayan retrieval corpus

## Metrics
| Metric | Value |
|---|---:|
| Bi-encoder Recall@10 | 0.0077 |
| Bi-encoder MRR@10 | 0.0026 |
| Reranked Recall@10 | 0.0308 |
| Reranked MRR@10 | 0.0067 |
| No-answer empty-correct | 20/20 |

## Slice metrics
| Slice | Reranked MRR@10 |
|---|---:|
| Arabic | 0.0033 |
| English | 0.0097 |
| Cross-lingual gap | 0.0063 |

## Behavioural tests
The Lab 6 classification behavioural suite is not directly applicable to the retrieval artefact.

## Known limitations
Strict relevant-ID retrieval metrics are very low and are difficult to interpret because the 20,000-row synthetic corpus contains 14,599 duplicate-text rows and only 5,401 unique texts. Judged relevant IDs are also concentrated in the early portion of the corpus. Topic hit@10 was perfect on the evaluated answerable queries, showing that semantically appropriate cases may be retrieved under different duplicate IDs; nevertheless, the strict-ID weakness and Arabic/English MRR gap remain important limitations.

## Contact / owner
Bayan course project team.