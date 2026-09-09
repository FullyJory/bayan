# Model Card — Bayan NER

## Intended use
Named-entity extraction from bilingual municipal feedback, including location entities.

## Artefact / data versions
- Model/checkpoint: xlm-roberta-base fine-tuned for NER
- Preprocessing version: Bayan Lab 3B / Lab 4 Arabic preprocessing
- Data version/snapshot: Bayan course NER dataset

## Metrics
| Metric | Value |
|---|---:|
| Test entity-level F1 | 1.0000 |
| Unsegmented LOCATION recall | 1.0000 |
| Segmented LOCATION recall | 1.0000 |

## Slice metrics
Lab 4 segmentation comparison showed no LOCATION-recall difference because both variants reached 1.0000.

## Behavioural tests
No separate NER behavioural suite was retained in Lab 6; the reported behavioural suite evaluates the topic classifier.

## Known limitations
The supplied NER benchmark exhibits a ceiling effect: both segmented and unsegmented LOCATION recall are 1.0000, so the evaluation cannot demonstrate the expected gain from Arabic clitic segmentation. These results come from the supplied course data and may overstate performance on noisier real-world Arabic, unseen dialects, novel entities, or more difficult boundary cases.

## Contact / owner
Bayan course project team.