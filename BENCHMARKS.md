# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | | | | | |
| XLM-R | | | | | |
| CAMeLBERT | | | | | |
| DistilBERT | | | | | |

- Golden preprocessing: ___ / 25 passed
- PII masking recall: ___ / 60 = ___%

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 1.4064 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 1.4064 |
| MARBERT (optional, not run) | — | — | — | — |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 0.0077 | 0.0026 | 15.45 ms |
| + cross-encoder rerank | 0.0308 | 0.0067 | 33.40 ms |
| cross-lingual slice | — | Arabic 0.0033 / English 0.0097 | — |

- no-answer empty-correct: 20 / 20
- cross-lingual gap: 0.0063
- retrieval targets achieved: No
- diagnosis: The 20,000-case synthetic corpus contains 14,599 duplicate-text rows (5,401 unique texts), while the judged relevant case IDs are concentrated in the first ~1,050 cases. The bi-encoder nevertheless achieved 130/130 topic hit@10, indicating semantically correct retrieval despite poor strict judged-ID recall caused by duplicate-case competition.

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | 70.0% | 100.0% |
| dialect-aware | | | | |

- Behavioural invariance: 140/200 passed (70.0%), below the approximate 95% course target.
- Minimum-functionality tests: 16/16 passed (100.0%), above the approximate 90% course target.
- Directional tests: not applicable to the Lab 3A topic classifier because the supplied directional skeletons require sentiment behaviour, while the classifier exposes topic labels only.
- Invariance failure analysis: ambiguous generic service statements changed between `digital_services` and `lighting` after otherwise irrelevant location/time substitutions, indicating sensitivity to contextual terms.
- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

### NER

- Model: xlm-roberta-base
- Task: Named Entity Recognition
- Evaluation: seqeval entity-level F1
- Validation F1: 1.0000
- Test Entity-level F1: 1.0000
- Training epochs: 3
- Learning rate: 2e-5
- Train batch size: 16
- Eval batch size: 32
- Saved artifact: artifacts/ner
- Target F1: >= 0.80
- Target achieved: Yes


### Lab 4 — NER Clitic Segmentation

- Base model: xlm-roberta-base
- Unsegmented LOCATION recall: 1.0000
- Segmented LOCATION recall: 1.0000
- LOCATION recall delta: +0.0000
- Segmentation scheme: CAMeL Tools d3tok
- Target improvement: approximately +4 recall points
- Target achieved: No (ceiling effect)
- Interpretation: The unsegmented baseline already achieved perfect LOCATION recall on the supplied dataset, leaving no room for a measurable recall improvement from clitic segmentation.


### Lab 4 — Arabic Bake-off Result

- Evaluation split: reproducible 80/20 split stratified by topic and dialect region
- Evaluation examples: 1,440 (960 Gulf; 480 MSA)
- CAMeLBERT-mix Gulf Macro-F1: 1.0000
- CAMeLBERT-DA Gulf Macro-F1: 1.0000
- Gulf Macro-F1 delta (DA vs mix): +0.0000
- Target improvement: >= +0.04 Gulf Macro-F1
- Target achieved: No (ceiling effect)
- Interpretation: Both Arabic-centric checkpoints achieved perfect Macro-F1 on all, Gulf, and MSA slices, so the supplied dataset does not provide headroom to demonstrate a dialect-aware performance gain.


### Lab 5 — L2 Normalisation Bug Diagnosis

- Correct L2-normalised bi-encoder Recall@10: 0.0077
- Correct L2-normalised bi-encoder MRR@10: 0.0026
- Intentionally unnormalised-index Recall@10: 0.0385
- Intentionally unnormalised-index MRR@10: 0.0068
- Unnormalised-index topic hit@10: 1.0000
- Embedding norm range (2,000-case sample): 2.4221–5.6439 (mean 4.0557; std 0.7075).
- Diagnosis: Omitting corpus L2 normalisation changes the retrieval geometry because raw inner product is affected by vector magnitude rather than cosine similarity. On this supplied duplicate-heavy synthetic benchmark, however, the planted failure did not produce a further strict-ID metric collapse: the unnormalised run scored slightly higher while topic hit@10 remained perfect. Therefore the failure cannot be approved or rejected by eyeballing plausible results; the labelled metrics must be inspected, and this experiment does not support claiming a normalisation-induced collapse for this checkpoint/data combination.
