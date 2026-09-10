# Bayan | بيان

### Bilingual Arabic–English Citizen Feedback Intelligence System

**Bayan (بيان)** is an end-to-end Natural Language Processing (NLP) system designed to understand, analyze, and retrieve information from **Arabic and English citizen feedback**.

The project combines multilingual preprocessing, Transformer-based language models, Arabic-aware NLP, Named Entity Recognition (NER), extractive Question Answering (QA), semantic search, systematic model evaluation, inference optimization, and API serving within a unified engineering pipeline.

Bayan was developed by **Jory Alshaalan** as part of **SDA-AIE-211 — Natural Language Processing with Transformers** at **SDAIA Academy**, part of the **Saudi Data & AI Authority (SDAIA)**.

---

## About Bayan

Citizen feedback is rarely clean or uniform. A single message may contain Arabic, English, dialectal expressions, spelling variations, personally identifiable information, informal language, emojis, or noisy formatting.

Bayan is designed to process this type of bilingual input and transform it into structured and useful information.

The system provides several complementary NLP capabilities:

- Arabic and English text preprocessing
- Personally Identifiable Information (PII) masking
- Topic classification
- Named Entity Recognition (NER)
- Extractive Question Answering
- Arabic-specific normalization
- Dialect-aware evaluation
- Bilingual semantic search
- Cross-encoder re-ranking
- Confidence-aware evaluation
- Behavioural testing
- Model cards and error analysis
- ONNX inference optimization
- INT8 quantization
- FastAPI-based model serving
- Production-oriented latency and load testing

Instead of treating these capabilities as separate experiments, Bayan connects them into a single NLP engineering workflow.

---

# System Architecture

```text
                 Raw Citizen Feedback
                  Arabic / English
                         │
                         ▼
              ┌─────────────────────┐
              │    Preprocessing    │
              │ Normalization + PII │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Transformer Models  │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   Classification       NER             QA
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Arabic-Aware NLP  │
              │ Dialect + Segments  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Semantic Search   │
              │ Bi-Encoder + FAISS  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Cross-Encoder       │
              │ Re-ranking          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Evaluation & Model  │
              │ Quality Analysis    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ ONNX / INT8         │
              │ Optimization        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   FastAPI Service   │
              └─────────────────────┘
```

---

# Core Capabilities

## 1. Bilingual Text Preprocessing

Bayan provides a shared preprocessing pipeline for **Arabic and English text**.

Real-world citizen feedback can contain inconsistent Unicode forms, repeated characters, unnecessary whitespace, Arabic tatweel, HTML remnants, emojis, code-switching, and sensitive personal information.

The preprocessing layer standardizes the input before it reaches downstream models while preserving useful linguistic information.

The pipeline handles:

- Unicode normalization
- Arabic and English text
- Arabic tatweel removal
- Repeated-character normalization
- Whitespace cleanup
- Code-switched Arabic/English input
- Emoji preservation
- HTML remnants
- Sentence segmentation

This shared preprocessing contract is reused across the system to reduce inconsistencies between model training and serving.

---

## 2. Privacy and PII Masking

Citizen feedback may contain sensitive information.

Bayan includes a dedicated **PII masking layer** that identifies and masks sensitive patterns before model processing.

Examples include:

- Phone numbers
- National-ID-shaped sequences
- Other structured personal identifiers covered by the preprocessing contract

The objective is to minimize unnecessary exposure of personal information while retaining the linguistic context required by downstream NLP models.

---

# Transformer Foundation

Bayan uses Transformer-based architectures as the foundation for its language understanding components.

The project also includes a lower-level implementation and analysis of Transformer attention mechanisms, including:

- Scaled dot-product attention
- Query, Key, and Value representations
- Multi-Head Attention
- Attention masking
- Causal masking
- Padding-mask behaviour
- Transformer parameter analysis

The attention mechanism follows:

```text
Q × Kᵀ
   │
   ▼
Scale by √dₖ
   │
   ▼
Apply Mask
   │
   ▼
Softmax
   │
   ▼
Attention Weights × V
```

Attention diagnostics are also used to investigate behaviours such as padding leakage and attention concentration.

---

# Tokenization and Multilingual Model Selection

Arabic and English have substantially different tokenization characteristics.

A tokenizer that performs efficiently for English may fragment Arabic words excessively, while an Arabic-specific tokenizer may perform poorly on English input.

Bayan therefore evaluates multiple Transformer tokenizers using measurable evidence.

The evaluated tokenizer families include:

- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- `distilbert-base-uncased`

The comparison considers:

- Arabic token fertility
- English token fertility
- Sequence lengths
- p95 sequence length

Based on the bilingual tokenizer audit, **XLM-RoBERTa (`xlm-roberta-base`)** was selected as the primary multilingual checkpoint because it provided a strong balance across both Arabic and English.

Measured tokenizer fertility:

| Tokenizer | Arabic Fertility | English Fertility |
|---|---:|---:|
| mBERT | 2.153 | 1.510 |
| XLM-R | 1.672 | 1.434 |
| CAMeLBERT | 1.405 | 2.705 |
| DistilBERT | 4.527 | 1.298 |

Measured p95 sequence lengths:

| Tokenizer | Arabic p95 | English p95 |
|---|---:|---:|
| mBERT | 25 | 23 |
| XLM-R | 19 | 21 |
| CAMeLBERT | 18 | 36 |
| DistilBERT | 45 | 19 |

These measurements support using XLM-R as a balanced bilingual model rather than selecting a checkpoint based only on popularity or aggregate performance.

---

# Topic Classification

Bayan includes a Transformer-based **topic classifier** for categorizing citizen feedback.

The classification pipeline supports bilingual Arabic and English input and assigns feedback to service-related categories.

The dataset contains topics including:

- Billing
- Digital services
- Licensing
- Lighting
- Parks
- Roads
- Waste
- Water

A traditional **TF-IDF + LinearSVC** model is used as a baseline before Transformer fine-tuning.

The dataset splitting strategy is group-aware, preventing feedback from the same citizen from leaking between training, validation, and test sets.

### Classification Results

The measured TF-IDF baseline achieved:

```text
Macro-F1: 1.0000
```

The fine-tuned XLM-R classifier also achieved:

```text
Test Macro-F1: 1.0000
```

Because the supplied dataset is highly template-like and contains strong lexical topic cues, the classical baseline already reached the metric ceiling. Consequently, the intended improvement target over the baseline could not be demonstrated numerically.

This limitation is documented rather than artificially modifying the evaluation protocol.

---

# Named Entity Recognition

Bayan includes a **Named Entity Recognition (NER)** system for extracting structured information from citizen feedback.

The NER pipeline addresses an important Transformer challenge: labels may originally exist at the word level while Transformer tokenizers split words into multiple subword tokens.

Bayan therefore implements explicit **BIO-label alignment** between words and subword tokens.

The implementation handles:

- Word-to-subword alignment
- BIO labels
- Non-first subword masking
- Arabic tokenization behaviour
- Entity-level evaluation

The NER model is evaluated using entity-level metrics through `seqeval`.

### NER Result

The trained NER system achieved:

```text
Entity-level F1: 1.0000
```

The project additionally evaluates Arabic segmentation and its effect on NER behaviour.

---

# Extractive Question Answering

Bayan includes an **extractive Question Answering (QA)** component.

Given a question and supporting context, the system identifies the most appropriate answer span from the context.

The QA post-processing logic validates candidate spans and prevents invalid selections.

A key design requirement is **honest no-answer behaviour**.

If the context does not contain a valid answer, Bayan can return:

```text
answer = None
```

instead of forcing an unsupported answer.

### QA Validation

The QA smoke evaluation includes both answerable and unanswerable examples.

Measured result:

```text
Answerable:
9 / 9 correct spans

Unanswerable:
3 / 3 correctly returned no answer
```

---

# Arabic-Aware NLP

Arabic requires language-specific processing beyond a generic multilingual pipeline.

Bayan includes a dedicated Arabic processing layer designed to investigate normalization, morphology, segmentation, and dialect variation.

---

## Arabic Normalization

The Arabic normalization component provides versioned normalization profiles.

It handles Arabic-specific orthographic variation while allowing the original display text to remain separate from the model-normalized representation.

The normalization implementation was validated against the supplied Arabic normalization contract:

```text
30 tests passed
```

---

## Dialect Analysis

Bayan does not assume that evaluation on Modern Standard Arabic is sufficient to represent Arabic performance.

The Arabic dataset was analyzed by dialect region.

Measured distribution:

```text
Arabic examples: 7,200

Gulf Arabic:
4,800 examples
66.7%

MSA:
2,400 examples
33.3%
```

This distribution makes dialect-specific evaluation important because aggregate Arabic metrics can hide performance differences between MSA and Gulf Arabic.

---

## Arabic Segmentation

The Arabic pipeline also incorporates morphological segmentation using CAMeL Tools.

Segmentation is evaluated as part of the NER pipeline to determine whether Arabic clitic handling improves entity recognition.

In the evaluated dataset, both segmented and unsegmented NER configurations reached the metric ceiling:

```text
Entity F1:        1.0000
LOCATION Recall:  1.0000
```

Therefore, the expected improvement from segmentation could not be demonstrated on this dataset.

---

## Arabic Model Comparison

Arabic-focused Transformer checkpoints were also compared across dialect slices.

The evaluation included:

- Overall Arabic
- Gulf Arabic
- Modern Standard Arabic

CAMeLBERT variants were evaluated to determine whether dialect specialization provided measurable benefits.

In the available benchmark, both evaluated configurations reached:

```text
All Arabic Macro-F1: 1.0000
Gulf Macro-F1:       1.0000
MSA Macro-F1:        1.0000
```

The result again demonstrates a ceiling effect in the supplied dataset rather than evidence that dialect variation is irrelevant.

---

# Bilingual Semantic Search

Bayan includes a semantic search engine for retrieving historical citizen cases related to an incoming query.

The search system uses a **two-stage retrieval architecture**.

```text
User Query
    │
    ▼
Multilingual Bi-Encoder
    │
    ▼
Query Embedding
    │
    ▼
FAISS Vector Search
    │
    ▼
Top Candidate Cases
    │
    ▼
Cross-Encoder
    │
    ▼
Re-ranked Results
```

---

## Bi-Encoder Retrieval

Bayan uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

to generate multilingual dense embeddings.

The historical corpus contains:

```text
20,000 cases
```

with:

```text
Embedding dimension: 384
```

Vectors are explicitly **L2-normalized** before being inserted into the FAISS index.

The index also stores a manifest containing information such as:

- Embedding model
- Preprocessing version
- Number of vectors
- Embedding dimension

This prevents incompatible preprocessing or model versions from being silently combined.

---

## Cross-Encoder Re-ranking

Initial semantic-search candidates are re-ranked using:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

The cross-encoder evaluates the query and candidate jointly to provide a more precise ranking than embedding similarity alone.

---

## No-Result Behaviour

Semantic retrieval systems should not always return an answer.

Bayan therefore includes a configurable minimum relevance score.

When no candidate exceeds the required threshold, the search service can return an empty result instead of presenting an unrelated historical case.

The no-answer evaluation achieved:

```text
20 / 20 correct
```

for the evaluated threshold range.

---

# Retrieval Evaluation

Bayan evaluates retrieval using labelled queries rather than approving results only because they appear semantically plausible.

The evaluation includes:

- Recall@10
- MRR@10
- Re-ranking comparison
- Arabic retrieval
- English retrieval
- Cross-lingual gap
- No-answer correctness
- Retrieval latency

Measured strict-ID results:

| Metric | Result |
|---|---:|
| Recall@10 without reranking | 0.0077 |
| MRR@10 without reranking | 0.0026 |
| Recall@10 with reranking | 0.0308 |
| MRR@10 with reranking | 0.0067 |
| Arabic reranked MRR | 0.0033 |
| English reranked MRR | 0.0097 |
| Cross-lingual MRR gap | 0.0063 |

The low strict-ID metrics were investigated rather than hidden.

The 20,000-case retrieval corpus contains only:

```text
5,401 unique texts
14,599 duplicate-text rows
```

The labelled relevance IDs are also concentrated in a relatively small portion of the corpus.

Despite the poor strict-ID retrieval metrics, topic-level retrieval achieved:

```text
Topic Hit@10: 130 / 130
```

This indicates an important distinction between retrieving a semantically/topically correct duplicate and retrieving the exact row ID expected by the benchmark.

The discrepancy is documented as an evaluation limitation rather than redefining the labels to improve reported performance.

---

# Retrieval Latency

Measured retrieval-stage latency:

```text
Bi-encoder retrieval p50:
15.45 ms / query

Cross-encoder re-ranking p50:
33.40 ms / query
```

These measurements provide separate visibility into first-stage retrieval and second-stage re-ranking costs.

---

# Evaluation Framework

Bayan includes a dedicated evaluation framework designed to measure more than aggregate model accuracy.

The framework combines:

- Bootstrap confidence intervals
- Slice-based evaluation
- Behavioural testing
- Manual error analysis
- Error taxonomy
- Model cards
- Known limitations

---

## Bootstrap Confidence Intervals

Bootstrap resampling is used to quantify uncertainty around evaluation metrics.

The classifier validation evaluation produced:

```text
Accuracy: 0.8750
95% Bootstrap CI: [0.8617, 0.8888]
```

The evaluation utilities also support paired bootstrap comparison when per-example predictions from both systems are available.

When only aggregate metrics are available, Bayan does not fabricate paired observations.

---

# Slice-Based Evaluation

Aggregate metrics can hide systematic failures.

Bayan therefore evaluates performance across multiple slices.

### Language

| Slice | Examples | Accuracy |
|---|---:|---:|
| Arabic | 1,200 | 75.0% |
| English | 1,200 | 100.0% |

### Dialect

| Slice | Examples | Accuracy |
|---|---:|---:|
| MSA | 1,200 | 75.0% |
| No dialect label | 1,200 | 100.0% |

### Input Length

| Slice | Examples | Accuracy |
|---|---:|---:|
| Short | 754 | 84.35% |
| Medium | 1,646 | 88.94% |

The evaluation identified a substantial Arabic/English performance difference that would not be visible from aggregate accuracy alone.

---

# Error Analysis

A manual review of **120 validation errors** was conducted.

All 120 reviewed failures followed the same systematic pattern:

```text
True class:      parks
Predicted class: roads
Language:        Arabic
```

The errors included explicit park-related cues such as references to:

- Parks
- Children's playgrounds
- Park irrigation
- Park maintenance
- Park walkways
- Accessibility

Because clean examples failed alongside noisier variants, the issue was classified as:

```text
Systematic class confusion
```

### Error Taxonomy

| Error Category | Count | Share |
|---|---:|---:|
| Systematic `parks → roads` confusion | 120 | 100% |

The analysis suggests prioritizing:

1. Auditing the training data, labels, and split distribution for the `parks` class.
2. Adding diverse Arabic park examples and hard negatives against the `roads` class.
3. Adding regression and behavioural tests for park-specific cues and noisy Arabic variants.

The exact improvement from these interventions must be measured experimentally rather than predicted as a guaranteed metric increase.

---

# Behavioural Evaluation

Bayan includes behavioural tests to investigate whether model predictions remain stable under controlled changes.

### Invariance Testing

Inputs are modified in ways that should not change the predicted topic, such as irrelevant location or time substitutions.

Measured result:

```text
140 / 200 passed
70%
```

The failures revealed instability between some `digital_services` and `lighting` predictions under irrelevant substitutions.

### Minimum Functionality Tests

A manually defined bilingual MFT set was used to verify basic topic recognition.

Measured result:

```text
16 / 16 passed
100%
```

### Directional Testing

The supplied directional tests depend on sentiment behaviour, while the trained Bayan artifact is a topic classifier.

These tests are therefore marked **not applicable** rather than reporting an artificial score.

---

# Model Cards

Bayan maintains model cards for three major components:

```text
docs/model_cards/topic_classifier.md
docs/model_cards/ner.md
docs/model_cards/retrieval.md
```

Each model card documents information such as:

- Intended use
- Model/checkpoint
- Evaluation evidence
- Known limitations
- Important performance slices
- Deployment considerations

The purpose is to make model limitations visible alongside model performance.

---

# Inference Optimization

Bayan extends beyond model training into production-oriented inference optimization.

The optimization ladder evaluates:

```text
PyTorch FP32
      │
      ▼
Dynamic Padding
      │
      ▼
Reduced Maximum Sequence Length
      │
      ▼
ONNX FP32
      │
      ▼
INT8 Quantization
```

Optimization decisions are based on both **latency and model quality**.

The evaluation considers:

- p50 inference latency
- p99 inference latency
- CPU thread configuration
- Speed-up relative to FP32
- Macro-F1
- Accuracy/quality tax
- Confidence intervals where applicable

The objective is to reduce inference cost without accepting an unjustified degradation in NLP quality.

---

# FastAPI Serving

The final Bayan architecture exposes the NLP pipeline through **FastAPI**.

The service is designed to integrate the same preprocessing contract used during development with the selected production model artifacts.

The integrated architecture supports endpoints such as:

```text
POST /v1/classify
POST /v1/entities
POST /v1/search
POST /v1/analyse
```

These endpoints represent the main Bayan capabilities:

- Classifying citizen feedback
- Extracting entities
- Retrieving related historical cases
- Combining multiple NLP operations into a unified analysis

---

# Startup Canaries

Model-serving failures can occur even when the API itself starts successfully.

Bayan therefore includes **startup canaries** designed to detect problems such as:

- Incompatible model artifacts
- Incorrect preprocessing versions
- Train/serve skew
- Unexpected model outputs
- Missing serving dependencies

The service verifies critical assumptions before accepting production traffic.

---

# Performance and Load Testing

Bayan distinguishes between **bare model latency** and **end-to-end HTTP latency**.

Model benchmarks use a representative production input-length mix from:

```text
data/serving/bench_mix.npy
```

The benchmark records:

- Warm-up behaviour
- p50 latency
- p99 latency
- CPU thread count
- Sequence-length configuration

The API is additionally designed to be load-tested using concurrent HTTP clients, allowing the complete serving path to be measured rather than relying only on isolated model inference.

---

# Repository Structure

```text
bayan/
│
├── src/
│   └── bayan/
│       │
│       ├── preprocessing/
│       │   ├── core.py
│       │   ├── segmentation.py
│       │   └── arabic.py
│       │
│       ├── attention.py
│       │
│       ├── models/
│       │   ├── data.py
│       │   ├── ner.py
│       │   └── qa.py
│       │
│       ├── search/
│       │   ├── index.py
│       │   └── service.py
│       │
│       ├── evaluation/
│       │   ├── bootstrap.py
│       │   ├── slices.py
│       │   └── behavioural.py
│       │
│       └── serving/
│           ├── api.py
│           └── canaries.py
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_tokenizer_audit.py
│   ├── 02_transformer_anatomy.py
│   └── 05_retrieval_eval.py
│
├── scripts/
│   ├── doctor.py
│   ├── parameter_audit.py
│   ├── tfidf_baseline.py
│   ├── train_classifier.py
│   ├── train_ner.py
│   ├── qa_smoke.py
│   ├── dialect_audit.py
│   ├── arabic_bakeoff.py
│   ├── evaluation_report.py
│   ├── benchmark_inference.py
│   ├── export_onnx.py
│   └── load_test.sh
│
├── tests/
├── data/
├── artifacts/
├── templates/
│   └── model_card.md.j2
│
├── docs/
│   ├── model_cards/
│   └── ERROR_TAXONOMY.md
│
├── NOTES.md
├── BENCHMARKS.md
├── DECISIONS.md
├── EVALUATION_REPORT.md
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

# Technologies

Bayan brings together tools from several areas of modern NLP and machine learning engineering.

### Machine Learning & NLP

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- scikit-learn
- spaCy
- CAMeL Tools
- seqeval

### Information Retrieval

- FAISS
- Multilingual sentence embeddings
- Bi-encoder retrieval
- Cross-encoder re-ranking

### Model Optimization

- ONNX
- ONNX Runtime
- INT8 quantization

### Serving & Testing

- FastAPI
- Uvicorn
- pytest
- HTTP load testing

### Data & Evaluation

- NumPy
- pandas
- Bootstrap confidence intervals
- Slice-based evaluation
- Behavioural testing
- Manual error analysis

---

# Engineering Principles

Bayan follows several principles throughout the project.

### Evidence Before Decisions

Model, tokenizer, retrieval, and optimization decisions are based on measured results rather than assumptions.

### Bilingual Evaluation

Arabic and English are evaluated separately where appropriate instead of relying exclusively on aggregate metrics.

### Honest Failure Reporting

Targets that are not achieved are documented rather than hidden or artificially satisfied.

### Privacy-Aware Processing

Sensitive information is masked before downstream NLP processing.

### No Forced Answers

Both QA and retrieval include explicit mechanisms for returning no answer or no result when the available evidence is insufficient.

### Reproducibility

Important model choices, benchmark results, evaluation findings, and limitations are recorded in the repository.

---

# Project Evidence

The repository maintains several documents that capture the engineering evidence behind Bayan.

### `BENCHMARKS.md`

Contains measured results including:

- Tokenizer fertility
- Sequence-length statistics
- Classification metrics
- NER metrics
- Arabic model comparisons
- Retrieval Recall@10 and MRR@10
- Behavioural-test results
- Inference latency
- Optimization quality tax

### `DECISIONS.md`

Documents evidence-based decisions such as:

- Tokenizer selection
- Multilingual checkpoint selection
- Arabic-model decisions
- Serving-artifact decisions
- Accepted engineering trade-offs

### `EVALUATION_REPORT.md`

Provides the broader model-quality analysis:

- Aggregate performance
- Confidence intervals
- Language and dialect slices
- Class-level performance
- Behavioural tests
- Error taxonomy
- Known limitations
- Prioritized improvements

### `docs/model_cards/`

Contains individual documentation for the major trained components.

---

# Known Limitations

Bayan is an engineering and educational NLP project, and its results should be interpreted in the context of the supplied datasets and evaluation environment.

Important limitations identified during development include:

- The topic-classification dataset contains strong lexical and template-like patterns, allowing both the traditional baseline and Transformer classifier to reach very high test performance.
- Validation evaluation revealed a systematic Arabic `parks → roads` classification failure.
- Arabic and English performance can differ substantially even when aggregate accuracy appears strong.
- Behavioural invariance testing exposed prediction instability under transformations that should ideally be irrelevant.
- The semantic-search corpus contains substantial duplicate text, making strict row-ID retrieval metrics significantly different from semantic/topic-level retrieval quality.
- Some desired paired statistical comparisons cannot be reconstructed when only aggregate historical results are available.
- Performance measurements depend on the execution environment and should not be generalized to different production hardware without re-benchmarking.

These limitations are retained as part of the project evidence rather than removed from the reported results.

---

# Future Improvements

Several directions could further improve Bayan:

- Expand the Arabic training corpus with more diverse real-world language.
- Increase representation of Gulf dialect varieties.
- Improve the `parks` class using targeted Arabic examples and hard negatives.
- Expand behavioural test coverage.
- Evaluate additional Arabic-focused Transformer models.
- Improve semantic-search relevance labels to account for duplicate or equivalent cases.
- Introduce more robust multilingual retrieval evaluation.
- Expand serving observability and monitoring.
- Evaluate inference performance across different CPU architectures.
- Extend the integrated API with additional analysis capabilities.

---

# Project Goal

Bayan demonstrates how a bilingual NLP system can be developed beyond simply fine-tuning a Transformer model.

It brings together the complete engineering lifecycle:

```text
Data
 ↓
Preprocessing
 ↓
Privacy Protection
 ↓
Tokenization
 ↓
Transformer Models
 ↓
Arabic-Aware NLP
 ↓
Classification / NER / QA
 ↓
Semantic Retrieval
 ↓
Evaluation
 ↓
Error Analysis
 ↓
Model Documentation
 ↓
Inference Optimization
 ↓
API Serving
```

The result is a unified **Arabic–English NLP engineering project** designed around measurable performance, transparent limitations, reproducibility, and production-oriented deployment.

---

# Author

**Jory Alshaalan**

Developed as part of:

**SDA-AIE-211 — Natural Language Processing with Transformers**

**SDAIA Academy**  
**Saudi Data & AI Authority (SDAIA)**

### SDAIA Academy

https://github.com/SDAIAAcademy
