from dataclasses import dataclass
import language_tool_python
import spacy

from metrics import (
    answer_type_score,
    cfg_score,
    content_density_score,
    dep_relevance_score,
    fluency_score,
    gazetteer_score,
    grammar_score,
    has_excessive_repetition,
    is_question_response,
    keyword_overlap_score,
    length_score,
    repetition_score,
    ttr_score,
)

WEIGHTS: dict[str, float] = {
    "length":          0.09,
    "repetition":      0.07,
    "keyword_overlap": 0.16,
    "fluency":         0.05,
    "dep_relevance":   0.11,
    "answer_type":     0.07,
    "grammar":         0.11,
    "cfg":             0.11,
    "ttr":             0.07,
    "content_density": 0.06,
    "gazetteer":       0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"


@dataclass
class EvaluationResult:
    question: str
    answer: str
    scores: dict[str, float]
    overall: float
    verdict: str   # "good" | "acceptable" | "poor"
    flags: list[str]  # override reasons, e.g. ["excessive_repetition"]


class QAEvaluator:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.tool = language_tool_python.LanguageTool("en-US")

    def evaluate(self, question: str, answer: str) -> EvaluationResult:
        q_doc = self.nlp(question)
        a_doc = self.nlp(answer)

        scores = {
            "length":          length_score(answer),
            "repetition":      repetition_score(answer),
            "keyword_overlap": keyword_overlap_score(q_doc, a_doc),
            "fluency":         fluency_score(a_doc),
            "dep_relevance":   dep_relevance_score(q_doc, a_doc),
            "answer_type":     answer_type_score(question, a_doc),
            "grammar":         grammar_score(answer, self.tool),
            "cfg":             cfg_score(a_doc),
            "ttr":             ttr_score(answer),
            "content_density": content_density_score(a_doc),
            "gazetteer":       gazetteer_score(question, answer),
        }

        overall = round(sum(scores[k] * WEIGHTS[k] for k in scores), 4)

        # Hard overrides: these are strong disqualifying signals
        flags: list[str] = []
        if has_excessive_repetition(answer):
            flags.append("excessive_repetition")
        if is_question_response(answer):
            flags.append("question_response")

        if flags:
            overall = round(min(overall, 0.35), 4)

        if overall >= 0.65:
            verdict = "good"
        elif overall >= 0.40:
            verdict = "acceptable"
        else:
            verdict = "poor"

        return EvaluationResult(
            question=question,
            answer=answer,
            scores=scores,
            overall=overall,
            verdict=verdict,
            flags=flags,
        )

    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[EvaluationResult]:
        return [self.evaluate(q, a) for q, a in pairs]
