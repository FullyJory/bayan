# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: Repeated characters
- Example: لووووسمحت
- Why it matters:Repeated characters add noise and increase the number of tokens without adding much meaning.
- Decision: clean 

### Defect 2
- Class: Emoji
- Example: 
- Why it matters: Emojis can express sentiment and user frustration, which may be useful for downstream classification.
- Decision: Preserve

### Defect 3
- Class: Personally identifiable information (PII)
- Example: 0551234567
- Why it matters: Phone numbers and national-ID-shaped values are sensitive information and should not be exposed to the model.
- Decision: mask 

### Defect 4
- Class: Tatweel
- Example: الطريـق
- Why it matters: Tatweel is a visual elongation that can create unnecessary token variations.
- Decision: Clean

### Defect 5
- Class: HTML remnants
- Example: <br>
- Why it matters: HTML markup is not part of the complaint meaning and can introduce unnecessary tokens.
- Decision: Clean 

### Defect 6
- Class: repeated words
- Example: My My Licence licence request
- Why it matters: Repeated words add noise and may increase the number of tokens without adding meaningful information.
- Decision: Clean

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution: Gulf — 4,800 (66.7%); MSA — 2,400 (33.3%); total Arabic rows — 7,200.
- One-sentence implication for MSA-only evaluation: Evaluating only on MSA would not represent the actual Arabic data distribution, because two-thirds of the Arabic slice is Gulf dialect, and could therefore overestimate performance on real Gulf-language inputs.
