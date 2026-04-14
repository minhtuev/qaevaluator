"""
Snorkel-based unsupervised QA evaluator.

Each metric from evaluator.py becomes a labeling function (LF) that votes
POOR / ACCEPTABLE / GOOD or ABSTAINs when it has no signal.
Snorkel's LabelModel learns the accuracy and correlations between LFs
and produces a stronger combined label — without using any ground-truth labels.
"""

import numpy as np
import pandas as pd
from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.analysis import get_label_buckets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

from evaluator import QAEvaluator
from gazetteer import find_matches
from test_pairs import TEST_PAIRS

ABSTAIN     = -1
POOR        =  0
ACCEPTABLE  =  1
GOOD        =  2

LABEL_NAMES = {POOR: "poor", ACCEPTABLE: "acceptable", GOOD: "good"}


# ── Build feature dataframe ───────────────────────────────────────────────────

def build_features(pairs: list[tuple[str, str]], evaluator: QAEvaluator) -> pd.DataFrame:
    rows = []
    for q, a in pairs:
        result = evaluator.evaluate(q, a)
        s = result.scores
        rows.append({
            "question":          q,
            "answer":            a,
            "length":            s["length"],
            "repetition":        s["repetition"],
            "keyword_overlap":   s["keyword_overlap"],
            "fluency":           s["fluency"],
            "dep_relevance":     s["dep_relevance"],
            "answer_type":       s["answer_type"],
            "grammar":           s["grammar"],
            "cfg":               s["cfg"],
            "ttr":               s["ttr"],
            "content_density":   s["content_density"],
            "excessive_rep":     int("excessive_repetition" in result.flags),
            "is_question":       int("question_response" in result.flags),
            "gazetteer":         s["gazetteer"],
            "gaz_has_match":     int(bool(find_matches(a))),
        })
    return pd.DataFrame(rows)


# ── Labeling functions ────────────────────────────────────────────────────────

def lf_too_short(x):
    if x.length < 0.2:   return POOR
    return ABSTAIN

def lf_good_length(x):
    if x.length == 1.0:  return GOOD
    return ABSTAIN

def lf_excessive_repetition(x):
    if x.excessive_rep:  return POOR
    return ABSTAIN

def lf_high_repetition(x):
    if x.repetition < 0.4:  return POOR
    return ABSTAIN

def lf_no_keyword_overlap(x):
    if x.keyword_overlap == 0.0:  return POOR
    return ABSTAIN

def lf_strong_keyword_overlap(x):
    if x.keyword_overlap >= 0.8:  return GOOD
    if x.keyword_overlap <= 0.3:  return POOR
    return ABSTAIN

def lf_no_dep_relevance(x):
    if x.dep_relevance == 0.0:    return POOR
    return ABSTAIN

def lf_strong_dep_relevance(x):
    if x.dep_relevance >= 0.8:    return GOOD
    return ABSTAIN

def lf_poor_fluency(x):
    if x.fluency < 0.4:  return POOR
    return ABSTAIN

def lf_poor_grammar(x):
    if x.grammar < 0.3:  return POOR
    if x.grammar < 0.6:  return ACCEPTABLE
    return ABSTAIN

def lf_good_grammar(x):
    if x.grammar == 1.0 and x.keyword_overlap > 0:  return GOOD
    return ABSTAIN

def lf_cfg_fail(x):
    if x.cfg == 0.0 and x.length == 1.0:  return POOR
    return ABSTAIN

def lf_cfg_pass(x):
    if x.cfg == 1.0:  return GOOD
    return ABSTAIN

def lf_question_response(x):
    if x.is_question:  return POOR
    return ABSTAIN

def lf_answer_type_mismatch(x):
    if x.answer_type == 0.3:  return POOR
    return ABSTAIN

def lf_answer_type_match(x):
    if x.answer_type == 1.0 and x.keyword_overlap > 0:  return GOOD
    return ABSTAIN

def lf_low_ttr(x):
    """Rambling answers repeat words heavily → low type-token ratio."""
    if x.ttr < 0.5 and x.length == 1.0:  return POOR
    return ABSTAIN

def lf_high_ttr(x):
    if x.ttr >= 0.8 and x.keyword_overlap > 0:  return GOOD
    return ABSTAIN

def lf_low_content_density(x):
    """Answers packed with filler rather than content words."""
    if x.content_density < 0.30 and x.length == 1.0:  return POOR
    return ABSTAIN

def lf_high_content_density(x):
    if x.content_density >= 0.55 and x.keyword_overlap > 0:  return GOOD
    return ABSTAIN

def lf_all_signals_good(x):
    """Fires GOOD only when multiple strong signals align."""
    if (x.keyword_overlap >= 0.5 and x.dep_relevance >= 0.5
            and x.grammar >= 0.8 and not x.excessive_rep and not x.is_question):
        return GOOD
    return ABSTAIN

def lf_all_signals_poor(x):
    """Fires POOR when keyword and dep relevance are both zero."""
    if x.keyword_overlap == 0.0 and x.dep_relevance == 0.0:
        return POOR
    return ABSTAIN

def lf_gazetteer_match_good(x):
    """Answer contains a named entity that aligns with the question type."""
    if x.gazetteer == 1.0:
        return GOOD
    return ABSTAIN

def lf_gazetteer_any_match(x):
    """Answer contains any gazetteer entity — better than pure gibberish."""
    if x.gaz_has_match and x.gazetteer >= 0.8:
        return GOOD
    return ABSTAIN

def lf_gazetteer_no_match_entity_question(x):
    """Question expects a named entity but answer has none."""
    if x.gazetteer == 0.2:   # only fires on wh-questions that expect an entity
        return POOR
    return ABSTAIN


LFS = [
    LabelingFunction(name="low_ttr",               f=lf_low_ttr),
    LabelingFunction(name="high_ttr",              f=lf_high_ttr),
    LabelingFunction(name="low_content_density",   f=lf_low_content_density),
    LabelingFunction(name="high_content_density",  f=lf_high_content_density),

    LabelingFunction(name="too_short",             f=lf_too_short),
    LabelingFunction(name="good_length",            f=lf_good_length),
    LabelingFunction(name="excessive_repetition",   f=lf_excessive_repetition),
    LabelingFunction(name="high_repetition",        f=lf_high_repetition),
    LabelingFunction(name="no_keyword_overlap",     f=lf_no_keyword_overlap),
    LabelingFunction(name="strong_keyword_overlap", f=lf_strong_keyword_overlap),
    LabelingFunction(name="no_dep_relevance",       f=lf_no_dep_relevance),
    LabelingFunction(name="strong_dep_relevance",   f=lf_strong_dep_relevance),
    LabelingFunction(name="poor_fluency",           f=lf_poor_fluency),
    LabelingFunction(name="poor_grammar",           f=lf_poor_grammar),
    LabelingFunction(name="good_grammar",           f=lf_good_grammar),
    LabelingFunction(name="cfg_fail",               f=lf_cfg_fail),
    LabelingFunction(name="cfg_pass",               f=lf_cfg_pass),
    LabelingFunction(name="question_response",      f=lf_question_response),
    LabelingFunction(name="answer_type_mismatch",   f=lf_answer_type_mismatch),
    LabelingFunction(name="answer_type_match",      f=lf_answer_type_match),
    LabelingFunction(name="all_signals_good",               f=lf_all_signals_good),
    LabelingFunction(name="all_signals_poor",               f=lf_all_signals_poor),
    LabelingFunction(name="gazetteer_match_good",           f=lf_gazetteer_match_good),
    LabelingFunction(name="gazetteer_any_match",            f=lf_gazetteer_any_match),
    LabelingFunction(name="gazetteer_no_match_entity_q",    f=lf_gazetteer_no_match_entity_question),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building features...")
    evaluator = QAEvaluator()
    pairs   = [(q, a) for q, a, _ in TEST_PAIRS]
    expected = [e for _, _, e in TEST_PAIRS]
    expected_int = [{"good": GOOD, "acceptable": ACCEPTABLE, "poor": POOR}[e] for e in expected]

    df = build_features(pairs, evaluator)

    # Apply all LFs to get the label matrix (n_examples x n_lfs)
    print("Applying labeling functions...")
    applier = PandasLFApplier(lfs=LFS)
    L = applier.apply(df)

    # ── LF coverage / conflict analysis ──────────────────────────────────────
    print("\n=== Labeling Function Analysis ===")
    from snorkel.labeling import LFAnalysis
    lf_summary = LFAnalysis(L=L, lfs=LFS).lf_summary()
    print(lf_summary.to_string())

    # ── Train label model (unsupervised — no ground truth used) ──────────────
    print("\nTraining LabelModel...")
    label_model = LabelModel(cardinality=3, verbose=False)
    label_model.fit(L_train=L, n_epochs=500, lr=0.01, seed=42)

    # ── Predict ───────────────────────────────────────────────────────────────
    preds = label_model.predict(L, tie_break_policy="abstain")
    probs = label_model.predict_proba(L)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n=== Predictions ===")
    print(f"{'#':<4} {'SNORKEL':<12} {'EXPECTED':<12} {'P(poor)':>8} {'P(acc)':>8} {'P(good)':>8}  ANSWER")
    print("─" * 100)
    for i, (pred, exp_int, exp_str, row) in enumerate(zip(preds, expected_int, expected, df.itertuples()), 1):
        pred_str  = LABEL_NAMES.get(pred, "abstain")
        match     = "✓" if pred == exp_int else "✗"
        ans_preview = row.answer[:55]
        print(f"{i:<4} {pred_str:<12} {exp_str:<12} {probs[i-1][0]:>8.3f} {probs[i-1][1]:>8.3f} {probs[i-1][2]:>8.3f}  {match} {ans_preview}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    valid = [(p, e) for p, e in zip(preds, expected_int) if p != ABSTAIN]
    correct = sum(p == e for p, e in valid)
    print(f"\nSnorkel accuracy (non-abstain): {correct}/{len(valid)} ({correct/len(valid)*100:.1f}%)")
    abstained = sum(1 for p in preds if p == ABSTAIN)
    print(f"Abstained: {abstained}/{len(preds)}")

    predicted_dist = Counter(LABEL_NAMES.get(p, "abstain") for p in preds)
    print(f"Predicted distribution: {dict(predicted_dist)}")
    print(f"Expected  distribution: {dict(Counter(expected))}")

    print("\nAvg prob by expected quality:")
    for quality, label_int in [("good", GOOD), ("acceptable", ACCEPTABLE), ("poor", POOR)]:
        group_probs = probs[[e == label_int for e in expected_int]]
        print(f"  {quality:<12}: P(poor)={group_probs[:,0].mean():.3f}  P(acc)={group_probs[:,1].mean():.3f}  P(good)={group_probs[:,2].mean():.3f}")

    # ── Semi-supervised bootstrap ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("=== Semi-Supervised Bootstrap (LogisticRegression on Snorkel pseudo-labels) ===")

    FEATURE_COLS = ["length", "repetition", "keyword_overlap", "fluency",
                    "dep_relevance", "answer_type", "grammar", "cfg",
                    "ttr", "content_density", "excessive_rep", "is_question",
                    "gazetteer", "gaz_has_match"]

    X = df[FEATURE_COLS].values
    confidence_threshold = 0.90
    max_probs = probs.max(axis=1)
    confident_mask = max_probs >= confidence_threshold
    pseudo_labels = preds[confident_mask]

    print(f"\nHigh-confidence examples (P >= {confidence_threshold}): {confident_mask.sum()}/{len(df)}")
    print(f"Pseudo-label distribution: {dict(Counter(LABEL_NAMES[l] for l in pseudo_labels))}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[confident_mask])
    X_all   = scaler.transform(X)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, pseudo_labels)
    bootstrap_preds = clf.predict(X_all)
    bootstrap_probs = clf.predict_proba(X_all)

    print("\n=== Bootstrap Predictions ===")
    print(f"{'#':<4} {'BOOTSTRAP':<12} {'EXPECTED':<12} {'P(poor)':>8} {'P(acc)':>8} {'P(good)':>8}  ANSWER")
    print("─" * 100)
    for i, (pred, exp_int, exp_str, row) in enumerate(zip(bootstrap_preds, expected_int, expected, df.itertuples()), 1):
        pred_str = LABEL_NAMES.get(pred, "abstain")
        match    = "✓" if pred == exp_int else "✗"
        print(f"{i:<4} {pred_str:<12} {exp_str:<12} {bootstrap_probs[i-1][0]:>8.3f} {bootstrap_probs[i-1][1]:>8.3f} {bootstrap_probs[i-1][2]:>8.3f}  {match} {row.answer[:55]}")

    print("\n" + "─" * 100)
    bs_correct = sum(p == e for p, e in zip(bootstrap_preds, expected_int))
    print(f"\nBootstrap accuracy: {bs_correct}/{len(bootstrap_preds)} ({bs_correct/len(bootstrap_preds)*100:.1f}%)")
    print(f"Snorkel  accuracy:  {correct}/{len(valid)} ({correct/len(valid)*100:.1f}%)")

    bs_dist = Counter(LABEL_NAMES.get(p) for p in bootstrap_preds)
    print(f"Bootstrap distribution: {dict(bs_dist)}")

    print("\nAvg bootstrap prob by expected quality:")
    for quality, label_int in [("good", GOOD), ("acceptable", ACCEPTABLE), ("poor", POOR)]:
        mask = [e == label_int for e in expected_int]
        gp = bootstrap_probs[mask]
        print(f"  {quality:<12}: P(poor)={gp[:,0].mean():.3f}  P(acc)={gp[:,1].mean():.3f}  P(good)={gp[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
