"""
Train and grid-search classifiers on the large QA dataset.

Steps
-----
1. Load JSONL dataset  (template or LLM — choose with --data)
2. Extract 14 features via QAEvaluator  (cached per dataset)
3. 60 / 20 / 20  train / val / test split  (stratified)
4. MindMeld-style grid search over four model families:
     LogReg · LinearSVC · RandomForest · GradientBoosting
   Scored by macro-F1 on StratifiedKFold(k=5) of training data
5. Report best model on val + test splits
6. Evaluate the winning model on the 93-pair TEST_PAIRS benchmark

Run:
    python train_classifier.py                        # use data/qa_large.jsonl
    python train_classifier.py --data data/qa_llm.jsonl
    python train_classifier.py --regen               # rebuild feature cache
"""

import sys
import pickle
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV

from evaluator import QAEvaluator
from gazetteer import find_matches
from test_pairs import TEST_PAIRS

LABEL_MAP   = {"poor": 0, "acceptable": 1, "good": 2}
LABEL_NAMES = {0: "poor", 1: "acceptable", 2: "good"}

FEATURE_COLS = [
    "length", "repetition", "keyword_overlap", "fluency",
    "dep_relevance", "answer_type", "grammar", "cfg",
    "ttr", "content_density", "gazetteer",
    "excessive_rep", "is_question", "gaz_has_match",
]

# ── MindMeld-style parameter grids ─────────────────────────────────────────────
# Reference: github.com/cisco/mindmeld text_models.py

PARAM_GRIDS = {
    "LogReg": {
        "model": LogisticRegression(max_iter=2000, random_state=42,
                                    class_weight="balanced", solver="saga"),
        "params": {
            "C":             [0.01, 0.1, 1, 10, 100, 1_000, 100_000],
            "penalty":       ["l1", "l2"],
            "fit_intercept": [True, False],
        },
    },
    "LinearSVC": {
        # wrapped in CalibratedClassifierCV to get predict_proba
        "model": CalibratedClassifierCV(
            LinearSVC(max_iter=5000, random_state=42, class_weight="balanced"),
            cv=3,
        ),
        "params": {
            "estimator__C": [0.01, 0.1, 1, 10, 100, 1_000],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "n_estimators": [50, 100, 200],
            "criterion":    ["gini", "entropy"],
            "max_depth":    [None, 5, 10],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators":  [50, 100, 200],
            "max_depth":     [3, 5],
            "learning_rate": [0.05, 0.1, 0.2],
        },
    },
}


# ── Feature extraction ──────────────────────────────────────────────────────────

def extract_features(pairs: list[tuple[str, str]],
                     evaluator: QAEvaluator) -> np.ndarray:
    try:
        from tqdm import tqdm
        it = tqdm(pairs, desc="Extracting features", unit="ex")
    except ImportError:
        it = pairs
    rows = []
    for q, a in it:
        r = evaluator.evaluate(q, a)
        s = r.scores
        rows.append([
            s["length"], s["repetition"], s["keyword_overlap"], s["fluency"],
            s["dep_relevance"], s["answer_type"], s["grammar"], s["cfg"],
            s["ttr"], s["content_density"], s["gazetteer"],
            float("excessive_repetition" in r.flags),
            float("question_response"    in r.flags),
            float(bool(find_matches(a))),
        ])
    return np.array(rows, dtype=float)


def load_or_build(jsonl_path: Path, regen: bool):
    cache_path = jsonl_path.with_suffix(".features.pkl")
    if cache_path.exists() and not regen:
        print(f"Loading cached features from {cache_path} …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Building features for {jsonl_path} …")
    data   = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    pairs  = [(d["question"], d["answer"]) for d in data]
    labels = np.array([LABEL_MAP[d["label"]] for d in data])

    ev = QAEvaluator()
    X  = extract_features(pairs, ev)

    payload = {"X": X, "y": labels}
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Cached → {cache_path}")
    return payload


# ── Reporting ───────────────────────────────────────────────────────────────────

def _report(name: str, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n── {name}  (n={len(y_true)}, acc={acc*100:.1f}%, macro-F1={mac:.3f}) ──")
    print(classification_report(
        y_true, y_pred,
        target_names=["poor", "acceptable", "good"],
        digits=3, zero_division=0,
    ))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix  (rows=true, cols=pred)  poor | acc | good")
    for i, row in enumerate(cm):
        print(f"  {['poor','acceptable','good'][i]:<12}: {row}")


# ── Grid search ─────────────────────────────────────────────────────────────────

def run_grid_search(X_train: np.ndarray, y_train: np.ndarray, n_jobs: int = -1):
    """Run MindMeld-style grid search.  Returns {name: (best_estimator, best_f1)}."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, spec in PARAM_GRIDS.items():
        t0 = time.time()
        print(f"\n  [{name}] searching {_grid_size(spec['params'])} combos …", end=" ", flush=True)
        gs = GridSearchCV(
            estimator  = spec["model"],
            param_grid = spec["params"],
            cv         = cv,
            scoring    = "f1_macro",
            n_jobs     = n_jobs,
            refit      = True,
            error_score= 0.0,
        )
        gs.fit(X_train, y_train)
        elapsed = time.time() - t0
        best_f1 = gs.best_score_
        print(f"best CV macro-F1={best_f1:.4f}  params={gs.best_params_}  ({elapsed:.1f}s)")
        results[name] = (gs.best_estimator_, best_f1)

    return results


def _grid_size(param_grid: dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/qa_large.jsonl",
                        help="Path to JSONL dataset (default: data/qa_large.jsonl)")
    parser.add_argument("--regen", action="store_true",
                        help="Rebuild feature cache from scratch")
    parser.add_argument("--no-gs", action="store_true",
                        help="Skip grid search (train default LogReg only)")
    args = parser.parse_args()

    jsonl_path = Path(args.data)
    if not jsonl_path.exists():
        sys.exit(f"ERROR: {jsonl_path} not found.")

    # ── Features ──────────────────────────────────────────────────────────────
    cache = load_or_build(jsonl_path, args.regen)
    X, y  = cache["X"], cache["y"]
    print(f"\nDataset : {jsonl_path.name}  |  {len(y):,} examples  |  {X.shape[1]} features")
    dist = Counter(LABEL_NAMES[i] for i in y)
    print(f"Labels  : {dict(sorted(dist.items()))}")

    # ── Split 60 / 20 / 20 ────────────────────────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)

    print(f"\nSplit   : train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Grid search ───────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("GRID SEARCH  (StratifiedKFold k=5, metric=macro-F1)")
    print("═" * 70)

    if args.no_gs:
        clf = LogisticRegression(C=1, max_iter=2000, random_state=42,
                                 class_weight="balanced")
        clf.fit(X_tr_s, y_train)
        gs_results = {"LogReg (default)": (clf, None)}
        best_name  = "LogReg (default)"
    else:
        gs_results = run_grid_search(X_tr_s, y_train)
        best_name  = max(gs_results, key=lambda k: gs_results[k][1])

    print(f"\n── Grid search summary ──")
    for name, (_, f1) in sorted(gs_results.items(), key=lambda kv: -(kv[1][1] or 0)):
        star = " ★ BEST" if name == best_name else ""
        if f1 is not None:
            print(f"  {name:<22}  CV macro-F1 = {f1:.4f}{star}")
        else:
            print(f"  {name:<22}  (default){star}")

    best_clf = gs_results[best_name][0]

    # ── Evaluate splits ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"SPLIT PERFORMANCE  — best model: {best_name}")
    print("═" * 70)
    _report("TRAIN", y_train, best_clf.predict(X_tr_s))
    _report("VAL",   y_val,   best_clf.predict(X_val_s))
    _report("TEST",  y_test,  best_clf.predict(X_test_s))

    # ── Feature importance (available for tree-based / logreg) ────────────────
    print("\n── Feature importances / coefficients ──")
    try:
        est = best_clf.estimator if hasattr(best_clf, "estimator") else best_clf
        if hasattr(est, "coef_"):
            coef = np.abs(est.coef_).mean(axis=0)
        elif hasattr(est, "feature_importances_"):
            coef = est.feature_importances_
        else:
            coef = None
        if coef is not None:
            for feat, val in sorted(zip(FEATURE_COLS, coef), key=lambda x: -x[1]):
                bar = "█" * int(val * 30)
                print(f"  {feat:<18} {val:.4f}  {bar}")
    except Exception:
        print("  (not available for this estimator)")

    # ── Per-model val comparison ───────────────────────────────────────────────
    print("\n── All-model val accuracy (quick comparison) ──")
    for name, (clf_i, _) in gs_results.items():
        preds_val = clf_i.predict(X_val_s)
        acc  = accuracy_score(y_val, preds_val)
        mac  = f1_score(y_val, preds_val, average="macro", zero_division=0)
        star = " ★" if name == best_name else ""
        print(f"  {name:<22}  acc={acc*100:.1f}%  macro-F1={mac:.3f}{star}")

    # ── Benchmark: 93-pair TEST_PAIRS ─────────────────────────────────────────
    print("\n" + "═" * 70)
    print("BENCHMARK: 93-pair TEST_PAIRS")
    print("═" * 70)

    bench_pairs = [(q, a) for q, a, _ in TEST_PAIRS]
    bench_y     = np.array([LABEL_MAP[e] for _, _, e in TEST_PAIRS])

    ev = QAEvaluator()
    X_bench   = extract_features(bench_pairs, ev)
    X_bench_s = scaler.transform(X_bench)

    print(f"\n{'#':<4} {'PREDICTED':<12} {'EXPECTED':<12}  ANSWER (55 chars)")
    print("─" * 85)
    bench_preds = best_clf.predict(X_bench_s)
    for i, (pred, exp_int, (q, a, exp_str)) in enumerate(
            zip(bench_preds, bench_y, TEST_PAIRS), 1):
        mark = "✓" if pred == exp_int else "✗"
        print(f"{i:<4} {LABEL_NAMES[pred]:<12} {exp_str:<12}  {mark} {a[:55]}")

    print()
    _report(f"BENCHMARK — {best_name}", bench_y, bench_preds)

    # ── Summary comparison ─────────────────────────────────────────────────────
    bench_acc = accuracy_score(bench_y, bench_preds)
    bench_mac = f1_score(bench_y, bench_preds, average="macro", zero_division=0)

    print("\n" + "═" * 70)
    print("ACCURACY COMPARISON  (93-pair benchmark)")
    print("═" * 70)
    rows = [
        ("Rule-based (run.py)",             54.8, None),
        ("Snorkel LabelModel",              64.5, None),
        ("Bootstrap pseudo-label LogReg",   65.6, None),
        ("Template-data LogReg",            66.7, None),
        (f"Grid-search {best_name} ←",      bench_acc * 100, bench_mac),
    ]
    for label, acc, mac in rows:
        bar  = "█" * int(acc / 2)
        tail = f"  macro-F1={mac:.3f}" if mac is not None else ""
        print(f"  {label:<40} {acc:5.1f}%  {bar}{tail}")

    print("\nPer-class accuracy on 93-pair benchmark:")
    for lstr, lint in [("good", 2), ("acceptable", 1), ("poor", 0)]:
        mask = bench_y == lint
        hit  = (bench_preds[mask] == lint).sum()
        n    = mask.sum()
        print(f"  {lstr:<12}  {hit}/{n}  ({hit/n*100:.1f}%)")


if __name__ == "__main__":
    main()
