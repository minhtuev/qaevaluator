from collections import Counter
from spacy.tokens import Doc
import language_tool_python
from nltk import CFG
from nltk.parse import EarleyChartParser

from gazetteer import gazetteer_score  # re-exported here for a single import surface

# Question type -> expected NER labels for answer-type matching
ANSWER_TYPE_MAP = {
    "who": {"PERSON", "ORG"},
    "when": {"DATE", "TIME"},
    "where": {"GPE", "LOC", "FAC"},
    "how many": {"CARDINAL", "QUANTITY"},
    "how much": {"CARDINAL", "MONEY", "QUANTITY"},
}


def length_score(answer: str, min_words: int = 3, max_words: int = 80) -> float:
    """Penalize answers that are too short or excessively long."""
    n = len(answer.split())
    if n < min_words:
        return round(n / min_words * 0.5, 4)
    if n > max_words:
        excess = n - max_words
        return round(max(0.0, 1.0 - (excess / max_words) * 0.5), 4)
    return 1.0


def repetition_score(answer: str) -> float:
    """Penalize repeated bigrams and trigrams."""
    words = answer.lower().split()
    if len(words) < 4:
        return 1.0

    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)]

    def repetition_ratio(ngrams: list[str]) -> float:
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values())
        return repeated / len(ngrams)

    penalty = (repetition_ratio(bigrams) + repetition_ratio(trigrams)) / 2
    return round(max(0.0, 1.0 - penalty * 2), 4)


def keyword_overlap_score(question_doc: Doc, answer_doc: Doc) -> float:
    """Lemmatized content-word overlap between question and answer."""
    q_keywords = {
        tok.lemma_.lower() for tok in question_doc
        if not tok.is_stop and not tok.is_punct
        and tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")
    }
    a_keywords = {
        tok.lemma_.lower() for tok in answer_doc
        if not tok.is_stop and not tok.is_punct
    }
    if not q_keywords:
        return 0.5
    return round(len(q_keywords & a_keywords) / len(q_keywords), 4)


def fluency_score(answer_doc: Doc) -> float:
    """
    Estimate grammatical completeness via dependency structure.
    Rewards sentences that have a root, a subject, and a verb.
    """
    sent_scores = []
    for sent in answer_doc.sents:
        has_root = any(tok.dep_ == "ROOT" for tok in sent)
        has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
        has_verb = any(tok.pos_ == "VERB" for tok in sent)
        sent_scores.append((has_root + has_subject + has_verb) / 3)
    return round(sum(sent_scores) / len(sent_scores), 4) if sent_scores else 0.0


def dep_relevance_score(question_doc: Doc, answer_doc: Doc) -> float:
    """
    Check whether content nouns from the question appear in syntactically
    prominent positions (subject / object / root) in the answer.
    A high score suggests the answer is actually about what was asked.
    """
    q_content = {
        tok.lemma_.lower() for tok in question_doc
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
    }
    if not q_content:
        return 0.5

    prominent_deps = {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "ROOT"}
    a_prominent = {
        tok.lemma_.lower() for tok in answer_doc
        if tok.dep_ in prominent_deps
    }
    return round(len(q_content & a_prominent) / len(q_content), 4)


def grammar_score(answer: str, tool: language_tool_python.LanguageTool) -> float:
    """
    Use LanguageTool to count grammar/spelling errors.
    Score decays with the number of errors relative to word count.
    """
    matches = tool.check(answer)
    words = len(answer.split())
    if words == 0:
        return 0.0
    error_rate = len(matches) / words
    return round(max(0.0, 1.0 - error_rate * 2), 4)


# Maps spaCy coarse POS tags to CFG terminal symbols
_POS_MAP = {
    "NOUN":  "N",
    "PROPN": "N",  # treat proper nouns as nouns for grammar purposes
    "VERB":  "V",
    "AUX":   "AUX",
    "DET":   "DET",
    "ADJ":   "ADJ",
    "ADP":   "P",
    "NUM":   "NUM",
    "ADV":   "ADV",
    "PRON":  "PRN",
}

# Simple English CFG over POS tag sequences
_GRAMMAR = CFG.fromstring("""
    S    -> NP VP | NP VP NP | NP VP PP | NP VP NP PP | NP AUX VP | NP AUX ADJ
    S    -> VP | VP NP | VP PP | NP VP ADJ
    NP   -> DET N | DET ADJ N | N | PRN | NP N | NP PP | NUM N | NUM | ADJ N | DET N N
    VP   -> V | V NP | V PP | V NP PP | AUX V | AUX V NP | V ADJ | AUX ADJ | AUX V PP
    PP   -> P NP | P PRN | P NUM
    DET  -> 'DET'
    N    -> 'N'
    V    -> 'V'
    AUX  -> 'AUX'
    ADJ  -> 'ADJ'
    P    -> 'P'
    NUM  -> 'NUM'
    ADV  -> 'ADV'
    PRN  -> 'PRN'
""")

_CFG_PARSER = EarleyChartParser(_GRAMMAR, trace=0)


def cfg_score(answer_doc: Doc) -> float:
    """
    Attempt to parse each sentence using a simple English CFG over POS tags.
    Sentences that yield at least one parse tree are considered structurally valid.
    Scrambled / ungrammatical word order typically produces no parse.
    Returns the fraction of sentences that parse successfully.
    """
    sent_scores = []
    for sent in answer_doc.sents:
        tags = [
            _POS_MAP[tok.pos_]
            for tok in sent
            if tok.pos_ in _POS_MAP and not tok.is_punct and not tok.is_space
        ]
        if not tags:
            continue
        try:
            parses = list(_CFG_PARSER.parse(tags))
            sent_scores.append(1.0 if parses else 0.0)
        except Exception:
            sent_scores.append(0.0)
    return round(sum(sent_scores) / len(sent_scores), 4) if sent_scores else 0.5


def ttr_score(answer: str) -> float:
    """
    Type-Token Ratio: unique_words / total_words.
    Penalizes rambling answers that repeat filler words heavily.
    Short answers (<5 words) get a neutral 0.5 to avoid penalising brevity twice.
    """
    words = answer.lower().split()
    if len(words) < 5:
        return 0.5
    return round(len(set(words)) / len(words), 4)


def content_density_score(answer_doc: Doc) -> float:
    """
    Ratio of content words (NOUN, PROPN, VERB, ADJ, ADV) to total tokens.
    Rambling answers are padded with filler; good answers pack more information per word.
    """
    tokens = [tok for tok in answer_doc if not tok.is_punct and not tok.is_space]
    if not tokens:
        return 0.0
    content = [tok for tok in tokens if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "ADV")]
    return round(len(content) / len(tokens), 4)


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "or", "but", "not", "it",
    "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "my", "your", "his", "her", "our", "their",
}


def has_excessive_repetition(answer: str, threshold: int = 3) -> bool:
    """
    Returns True if any content-word unigram or multi-word n-gram appears
    >= threshold times.  Stop-word-only unigrams are skipped so common
    function words ("the", "is", …) don't trigger false positives on normal
    answers.  Bigrams and trigrams are checked regardless of content because
    a repeated phrase is always degenerate.
    """
    words = answer.lower().split()

    # Unigrams: only count content words
    content_words = [w for w in words if w.rstrip(".,!?;:") not in _STOP_WORDS]
    if content_words:
        top_count = Counter(w.rstrip(".,!?;:") for w in content_words).most_common(1)[0][1]
        if top_count >= threshold:
            return True

    # Bigrams and trigrams: any repetition counts
    for n in (2, 3):
        ngrams = [
            "_".join(words[i:i + n])
            for i in range(len(words) - n + 1)
        ]
        if ngrams and Counter(ngrams).most_common(1)[0][1] >= threshold:
            return True

    return False


def is_question_response(answer: str) -> bool:
    """
    Returns True if the answer is itself a question — a strong signal
    of an evasive or non-answer response.
    """
    return answer.strip().endswith("?")


def answer_type_score(question: str, answer_doc: Doc) -> float:
    """
    For wh-questions, check whether the answer contains the expected
    named-entity type (e.g. PERSON for 'who', DATE for 'when').
    """
    q_lower = question.lower().strip()
    for wh, expected_labels in ANSWER_TYPE_MAP.items():
        if q_lower.startswith(wh):
            found_labels = {ent.label_ for ent in answer_doc.ents}
            return 1.0 if found_labels & expected_labels else 0.3
    return 0.75  # neutral for non-wh questions
