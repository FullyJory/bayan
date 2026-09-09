# Model Card — Bayan Topic Classifier

## Intended use
Bilingual Arabic/English municipal-feedback topic classification across eight Bayan service categories.

## Artefact / data versions
- Model/checkpoint: xlm-roberta-base fine-tuned for 8 topic labels
- Preprocessing version: Bayan Lab 3A topic-classification pipeline
- Data version/snapshot: data/raw/bayan_feedback.csv

## Metrics
| Metric | Value | 95% CI |
| --- | --- | --- |
| Validation accuracy | 87.50% | [86.17%, 88.88%] |
| Lab 3A test Macro-F1 | 1.0000 | not retained |

## Slice metrics
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

## Behavioural tests
| Test | Result | Target | Status |
| --- | --- | --- | --- |
| Invariance | 140/200 (70.0%) | approximately >=95% | below target |
| Minimum functionality | 16/16 (100.0%) | approximately >=90% | meets target |
| Directional | N/A | N/A | sentiment behaviour unsupported by topic-only artefact |

## Known limitations
The validation evidence is highly uneven across language and class slices. Arabic accuracy is 75.0%, and the `parks` class has 0.0% accuracy because all 300 validation `parks` examples are predicted as `roads`. Invariance testing also reaches only 70.0%, indicating sensitivity to otherwise irrelevant contextual substitutions. The dataset is synthetic/template-like, so perfect Lab 3A test Macro-F1 should not be interpreted as evidence of equivalent real-world performance.

## Contact / owner
Bayan course project team.