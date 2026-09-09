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
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

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
