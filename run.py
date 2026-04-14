from collections import Counter

from evaluator import QAEvaluator
from test_pairs import TEST_PAIRS


def main() -> None:
    evaluator = QAEvaluator()
    pairs = [(q, a) for q, a, _ in TEST_PAIRS]
    expected = [e for _, _, e in TEST_PAIRS]
    results = evaluator.evaluate_batch(pairs)

    # ── Per-result output ─────────────────────────────────────────────────────
    print(f"{'#':<4} {'VERDICT':<12} {'EXPECTED':<12} {'SCORE':<7}  "
          f"{'len':>5} {'rep':>5} {'kw':>5} {'flu':>5} {'dep':>5} {'type':>5} {'gram':>5} {'cfg':>5} {'gaz':>5}  "
          f"ANSWER (truncated)")
    print("─" * 140)

    for i, (result, exp) in enumerate(zip(results, expected), 1):
        s = result.scores
        match = "✓" if result.verdict == exp else "✗"
        answer_preview = result.answer[:55].replace("\n", " ")
        flags_display = f"[{','.join(result.flags)}]" if result.flags else ""
        print(
            f"{i:<4} {result.verdict:<12} {exp:<12} {result.overall:<7.3f}  "
            f"{s['length']:>5.2f} {s['repetition']:>5.2f} {s['keyword_overlap']:>5.2f} "
            f"{s['fluency']:>5.2f} {s['dep_relevance']:>5.2f} {s['answer_type']:>5.2f} "
            f"{s['grammar']:>5.2f} {s['cfg']:>5.2f} {s['gazetteer']:>5.2f}  "
            f"{match} {flags_display} {answer_preview}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 120)
    predicted = Counter(r.verdict for r in results)
    actual = Counter(expected)

    print("\nPredicted distribution:", dict(predicted))
    print("Expected  distribution:", dict(actual))

    correct = sum(1 for r, e in zip(results, expected) if r.verdict == e)
    print(f"\nVerdict accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

    # ── Score breakdown by expected quality ──────────────────────────────────
    print("\nAvg overall score by expected quality:")
    for quality in ("good", "acceptable", "poor"):
        group = [r.overall for r, e in zip(results, expected) if e == quality]
        if group:
            print(f"  {quality:<12}: {sum(group)/len(group):.3f}  (n={len(group)})")


if __name__ == "__main__":
    main()
