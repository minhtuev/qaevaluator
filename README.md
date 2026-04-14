# QA Evaluator

A research project exploring multiple strategies for classifying QA answer quality into three tiers: **good**, **acceptable**, and **poor** — without any reference answers.

Evaluated on a fixed 93-pair benchmark (`test_pairs.py`) across all approaches.

---

## Strategies

### 1. Rule-Based Heuristics (`run.py` + `evaluator.py`)

A hand-tuned weighted scorer combining 11 surface and linguistic signals:

| Feature | Description |
|---|---|
| `length` | Penalizes too-short or excessively long answers |
| `repetition` | Token-level repetition ratio |
| `keyword_overlap` | Question keyword coverage in the answer |
| `fluency` | Log-probability proxy via token frequency |
| `dep_relevance` | spaCy dependency parse alignment between Q and A |
| `answer_type` | Detects expected entity type (capital, person, date…) |
| `grammar` | LanguageTool error density |
| `cfg` | CFG parse depth as a complexity/structure signal |
| `ttr` | Type-token ratio (lexical diversity) |
| `content_density` | Ratio of content words to total words |
| `gazetteer` | Named-entity match against a 340-entry gazetteer (14 categories) |

Hard overrides cap the score at 0.35 for answers that trigger `excessive_repetition` or are themselves questions (`question_response` flag). The repetition check skips stop words to avoid false positives on common function words.

**Benchmark accuracy: 54.8%**

---

### 2. Snorkel Weak Supervision (`snorkel_model.py`)

Uses [Snorkel](https://snorkel.ai) to combine ~25 noisy labeling functions (LFs) into a probabilistic `LabelModel` — no hand-labeled data needed.

LF categories:
- Score thresholds (e.g., `lf_low_grammar`, `lf_high_keyword_overlap`)
- Flag-based rules (e.g., `lf_excessive_repetition`, `lf_is_question`)
- Combined multi-signal rules (e.g., `lf_very_short_and_low_kw`, `lf_high_all_three`)
- Gazetteer signals (category-aligned match → GOOD; expected but missing → POOR)

The LabelModel learns LF weights and correlations via expectation-maximization, then outputs soft probability labels.

**Key limitation**: the LabelModel cannot implement hard overrides — when many LFs vote GOOD (keyword overlap, grammar, dep_relevance all fire for a repetitive-but-relevant answer), one POOR-voting LF is outvoted regardless of its reliability weight.

**Benchmark accuracy: 64.5%**

---

### 3. Bootstrap Pseudo-Label Classifier (`snorkel_model.py`)

Extends Snorkel: the LabelModel's high-confidence predictions are used as pseudo-labels to train a supervised LogisticRegression on the same 14 features.

- Threshold: keep examples where `max(P(good), P(acceptable), P(poor)) ≥ 0.85`
- Features: all 11 evaluator scores + `excessive_rep`, `is_question`, `gaz_has_match` flags
- Addresses the LabelModel's inability to learn feature interactions

**Benchmark accuracy: 65.6%**

---

### 4. Supervised Classifier — Template Data (`train_classifier.py`)

Generated 5,250 labeled examples from 175 factual seed pairs using deterministic templates (`generate_dataset.py`). No external API required.

Each seed yields 3 question variants × 10 answers = 30 examples:
- **3 good**: complete, specific, varied phrasing
- **2 acceptable**: correct but vague, missing the specific entity
- **5 poor**: single-word, off-topic, repetitive, scrambled, evasive

**Training**: MindMeld-style grid search over 4 model families with `StratifiedKFold(k=5)`, scoring by macro-F1:

| Model | CV macro-F1 |
|---|---|
| GradientBoosting (lr=0.2, depth=3, n=200) | **0.969** |
| RandomForest (entropy, depth=10) | 0.965 |
| LogReg (C=100, l2) | 0.860 |
| LinearSVC (C=10) | 0.853 |

**Benchmark accuracy: 68.8%** (GradientBoosting) — best result overall, but the template distribution is narrow and overfits to its own patterns (97.5% on held-out template test data vs. 68.8% on real pairs).

---

### 5. Supervised Classifier — Manual Data (`train_classifier.py`)

200 hand-authored examples across 13 diverse factual seeds (`data/qa_manual.jsonl`), written to cover naturalistic variation that templates miss.

Seeds: speed of light, Cleopatra, earthquakes, Hamlet, photosynthesis, Amazon River, fall of Rome, DNA, printing press, boiling point of water, largest ocean, Sistine Chapel, gravity.

Each seed: 5 good + 3 acceptable + 7 poor (7 distinct subtypes: single-word, repetitive, scrambled, off-topic, evasive-question, wrong-entity, rambling).

**Grid search winner: RandomForest** (entropy, max_depth=5, n_estimators=50), CV macro-F1=0.712

**Benchmark accuracy: 66.7%** — slightly below template data despite higher quality, due to 26× smaller training size.

---

### 6. LLM-Generated Data (`generate_dataset_llm.py`)

Uses Claude Haiku to generate naturalistic answer variation for 250+ factual seeds. Each API call produces 15 examples per seed (5 good, 3 acceptable, 7 poor with distinct subtypes), targeting ~3,750 total examples.

Requires `ANTHROPIC_API_KEY`. Supports `--resume` to continue partial runs.

**Status**: dataset generation infrastructure complete; training results pending API access.

---

## Benchmark Summary

All results on the fixed 93-pair `TEST_PAIRS` benchmark:

```
Strategy                              Accuracy   Notes
────────────────────────────────────  ─────────  ──────────────────────────────────
Rule-based heuristics                   54.8%    No training data
Snorkel LabelModel                      64.5%    ~25 labeling functions, no labels
Bootstrap pseudo-label LogReg           65.6%    Snorkel → pseudo-labels → LogReg
Manual-data RandomForest (200 ex)       66.7%    Hand-crafted diverse examples
Template-data GradientBoosting (5k ex)  68.8%    Best overall; narrow distribution
LLM-generated data                      TBD      Pending API key
```

### Per-class accuracy (best model, template GradientBoosting)

| Class | Accuracy | Note |
|---|---|---|
| good | ~77% | Easiest to detect |
| poor | ~74% | Strong signal from flags + low scores |
| acceptable | ~35% | Hard ceiling across all methods |

### Key finding: the acceptable ceiling

The **acceptable/good boundary is the fundamental bottleneck** across every approach. Acceptable answers are fluent, grammatically correct, topically relevant, and often contain question keywords — they just omit the specific named entity. Without a reference answer or semantic grounding, surface and linguistic features alone cannot reliably distinguish "correct but vague" from "correct and specific." This ceiling (~35% recall on acceptable) was consistent across all five strategies.

---

## Running

```bash
source venv/bin/activate

# Strategy 1: rule-based
python run.py

# Strategy 2+3: Snorkel + bootstrap
python snorkel_model.py

# Generate template dataset (no API needed)
python generate_dataset.py

# Generate LLM dataset (needs ANTHROPIC_API_KEY)
python generate_dataset_llm.py --limit 20   # smoke test
python generate_dataset_llm.py              # full run

# Train + grid search on any dataset
python train_classifier.py                          # template data (default)
python train_classifier.py --data data/qa_manual.jsonl
python train_classifier.py --data data/qa_llm.jsonl
python train_classifier.py --regen                  # rebuild feature cache
python train_classifier.py --no-gs                  # skip grid search
```

## File Overview

```
evaluator.py          Weighted scorer; hard overrides for repetition/question flags
metrics.py            Individual scoring functions (length, repetition, fluency, etc.)
gazetteer.py          340-entry named-entity gazetteer with category-aligned scoring
run.py                Runs rule-based evaluator on the 93-pair benchmark
snorkel_model.py      Snorkel LabelModel + bootstrap LogReg
generate_dataset.py   Template-based 5k example generator (no API)
generate_dataset_llm.py  LLM-based generator via Claude Haiku
train_classifier.py   Grid search + evaluation pipeline for supervised classifiers
test_pairs.py         93 hand-labeled QA pairs used as the fixed benchmark
data/                 Generated datasets + feature caches (.features.pkl)
```
