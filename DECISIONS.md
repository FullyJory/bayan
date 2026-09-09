# Decision Records

## tokenizer
- Chosen checkpoint(s):
- Arabic fertility evidence:
- English fertility evidence:
- p95 length evidence:
- Operational trade-off / rationale:

## arabic-model
- Incumbent: CAMeLBERT-mix (`CAMeL-Lab/bert-base-arabic-camelbert-mix`)
- Candidate: CAMeLBERT-DA (`CAMeL-Lab/bert-base-arabic-camelbert-da`)
- All/Gulf/MSA evidence: CAMeLBERT-mix = 1.0000 / 1.0000 / 1.0000; CAMeLBERT-DA = 1.0000 / 1.0000 / 1.0000 Macro-F1.
- CI-backed verdict: No performance-based switch is justified from this bake-off; both checkpoints reached the evaluation ceiling, with a Gulf delta of +0.0000. Confidence intervals are deferred to Lab 6.
- Segmentation contract: Arabic NER uses CAMeL Tools d3tok clitic segmentation consistently; segmented and unsegmented LOCATION recall were both 1.0000 on the supplied NER data.

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
