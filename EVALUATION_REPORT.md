# EVALUATION REPORT — Bayan

## Executive headline
Validation performance is strongly uneven across slices: English accuracy is 100.0% while Arabic accuracy is 75.0%, with short inputs also performing below medium-length inputs (84.4% vs. 88.9%). The most critical quality risk is the parks class, which has 0.0% accuracy across 300 validation examples and should be prioritised for error analysis and remediation.

## Aggregate metric with bootstrap CI
Validation accuracy is **87.50%**, with a 95% percentile bootstrap confidence interval of **[86.17%, 88.88%]** based on 2,000 bootstrap resamples.

The retained Lab 4 Arabic bake-off reports CAMeLBERT-mix and CAMeLBERT-DA at identical Macro-F1 of 1.0000, giving an observed aggregate difference of 0.0000. A paired bootstrap interval cannot be computed honestly from the retained `results.csv` because it contains aggregate metrics only rather than paired per-example predictions.

## Sliced metrics with bootstrap CIs
| Slice | n | Accuracy | 95% CI | Small slice |
| --- | --- | --- | --- | --- |
| language=ar | 1200 | 75.00% | [72.58%, 77.58%] | no |
| language=en | 1200 | 100.00% | [100.00%, 100.00%] | no |
| dialect=MSA | 1200 | 75.00% | [72.58%, 77.58%] | no |
| dialect=missing | 1200 | 100.00% | [100.00%, 100.00%] | no |
| class=billing | 300 | 100.00% | [100.00%, 100.00%] | no |
| class=digital_services | 300 | 100.00% | [100.00%, 100.00%] | no |
| class=licensing | 300 | 100.00% | [100.00%, 100.00%] | no |
| class=lighting | 300 | 100.00% | [100.00%, 100.00%] | no |
| class=parks | 300 | 0.00% | [0.00%, 0.00%] | no |
| class=roads | 300 | 100.00% | [100.00%, 100.00%] | no |
| class=waste | 300 | 100.00% | [100.00%, 100.00%] | no |
| class=water | 300 | 100.00% | [100.00%, 100.00%] | no |
| length=medium | 1646 | 88.94% | [87.42%, 90.52%] | no |
| length=short | 754 | 84.35% | [81.70%, 86.87%] | no |

No reported slice is below the configured small-slice threshold of 20 examples.

## Behavioural suite
| Test | Result | Target | Status |
| --- | --- | --- | --- |
| Invariance | 140/200 (70.0%) | approximately >=95% | below target |
| Minimum functionality | 16/16 (100.0%) | approximately >=90% | meets target |
| Directional | N/A | N/A | sentiment behaviour unsupported by topic-only artefact |

The invariance failures were concentrated in ambiguous generic service statements whose predicted topic changed between `digital_services` and `lighting` after location/time substitutions. The measured 70.0% invariance rate is reported as-is rather than modifying the test to force the benchmark target.

## Error taxonomy
A reproducible sample of **120 validation errors was hand-read**. All 120 were Arabic `parks` examples predicted as `roads`, producing the following primary-error histogram:

| Primary error category | Count | Share |
|---|---:|---:|
| Systematic class confusion (`parks` -> `roads`) | 120 | 100.0% |

Some reviewed examples also contained spelling variation, elongation, emoji, or masked PII, but clean versions of the same patterns failed as well; these were therefore treated as secondary characteristics rather than the primary cause.

### Top 3 prioritised fixes

1. **Audit the `parks` training/split pipeline and label distribution.** Predicted impact: potentially large because all 300 observed validation errors belong to the same `parks` -> `roads` failure mode; the exact delta must be measured after retraining.
2. **Add diverse Arabic `parks` examples and hard negatives against `roads`.** Predicted impact: improvement primarily to `parks` recall and Arabic aggregate accuracy; exact delta requires a controlled retraining experiment.
3. **Add regression tests for park cues and noisy Arabic variants.** Predicted impact: primarily prevention of recurrence rather than a defensible immediate metric gain.

No numeric improvement is fabricated for these proposed fixes; predicted deltas require post-fix evaluation.

## Retrieval quality
The Lab 5 two-stage retrieval evaluation produced strict-ID Recall@10 of **0.0077** and MRR@10 of **0.0026** for the bi-encoder, increasing to Recall@10 **0.0308** and MRR@10 **0.0067** after cross-encoder reranking.

Arabic reranked MRR@10 was **0.0033** versus **0.0097** for English, a cross-lingual gap of **0.0063**. The corpus is duplicate-heavy: 20,000 rows contain only 5,401 unique texts, so strict relevant-ID retrieval substantially understates semantic topic retrieval; topic hit@10 was 130/130.

## Known limitations
- The validation predictions show a severe Arabic `parks` -> `roads` class-confusion failure: `parks` accuracy is 0.0% across 300 examples.
- Behavioural invariance is only 70.0%, below the approximate 95% course target.
- The supplied directional behavioural skeletons require sentiment behaviour, but the evaluated Lab 3A artefact exposes topic labels only, so those tests are not applicable.
- The supplied evaluation data is synthetic/template-like and contains repeated patterns, limiting how confidently these metrics transfer to real municipal feedback.
- Retrieval strict-ID metrics are distorted by extensive duplicate texts and concentrated relevance IDs.
- The retained Arabic bake-off artifact contains aggregate scores only, so a paired bootstrap comparison between its two checkpoints cannot be reconstructed without rerunning per-example inference.
