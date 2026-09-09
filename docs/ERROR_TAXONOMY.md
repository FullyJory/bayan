# Error Taxonomy

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect
9. Systematic class confusion

## Lab 6 Manual Error Review

A reproducible sample of 120 validation errors was hand-read.

### Primary error histogram

| Error category | Count | Share |
|---|---:|---:|
| Systematic class confusion (`parks` → `roads`) | 120 | 100.0% |

All 120 reviewed errors were Arabic `parks` examples predicted as `roads`.
The examples contained explicit park-related cues such as `حديقة`, children's
playground maintenance, park irrigation, and park walkway accessibility.

Some examples contained secondary surface noise such as spelling variation,
elongation, emoji, or masked PII tokens. However, clean versions of the same
templates were also misclassified, so these were not assigned as the primary
error cause.

### Top 3 prioritised fixes

1. Audit the training/split pipeline and label distribution for the `parks`
   class to identify why the classifier systematically maps it to `roads`.
2. Add diverse Arabic `parks` training examples and hard negatives contrasting
   park walkways with ordinary road complaints.
3. Add regression and behavioural tests for explicit park cues and noisy Arabic
   variants so the `parks` → `roads` failure cannot silently recur.

### Predicted metric impact

The 300 validation errors are all `parks` → `roads`. Therefore fixing this
systematic class confusion has substantially greater expected impact than
optimising isolated spelling or preprocessing variants. Exact metric deltas
should be measured after retraining rather than claimed from the manual review
alone.
