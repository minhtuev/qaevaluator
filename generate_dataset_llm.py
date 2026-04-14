"""
LLM-based QA dataset generator using Claude Haiku.

For each factual seed (question + correct fact), Claude generates answer
examples at three quality levels — producing naturalistic variation that
template strings cannot achieve.

Run:
    python generate_dataset_llm.py              # full run
    python generate_dataset_llm.py --limit 20  # quick smoke-test (20 seeds)
    python generate_dataset_llm.py --resume     # continue a partial run

Output: data/qa_llm.jsonl
        data/qa_llm_progress.jsonl  (auto-removed when complete)

Requires: ANTHROPIC_API_KEY env var
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import anthropic

Path("data").mkdir(exist_ok=True)
OUT_PATH      = Path("data/qa_llm.jsonl")
PROGRESS_PATH = Path("data/qa_llm_progress.jsonl")

MODEL = "claude-haiku-4-5-20251001"

# ── Factual seeds (question, correct_fact) ─────────────────────────────────────

# country-capital seeds
_COUNTRY_CAPITAL = [
    ("France","Paris","Western Europe"),("Germany","Berlin","Central Europe"),
    ("Japan","Tokyo","East Asia"),("China","Beijing","East Asia"),
    ("Brazil","Brasília","South America"),("Canada","Ottawa","North America"),
    ("United States","Washington","North America"),("Italy","Rome","Southern Europe"),
    ("Spain","Madrid","Southern Europe"),("Australia","Canberra","Oceania"),
    ("Russia","Moscow","Eastern Europe"),("India","New Delhi","South Asia"),
    ("Mexico","Mexico City","North America"),("Argentina","Buenos Aires","South America"),
    ("Egypt","Cairo","North Africa"),("South Africa","Pretoria","Southern Africa"),
    ("South Korea","Seoul","East Asia"),("Turkey","Ankara","Western Asia"),
    ("Saudi Arabia","Riyadh","the Middle East"),("Nigeria","Abuja","West Africa"),
    ("Poland","Warsaw","Central Europe"),("Netherlands","Amsterdam","Western Europe"),
    ("Sweden","Stockholm","Scandinavia"),("Norway","Oslo","Scandinavia"),
    ("Denmark","Copenhagen","Scandinavia"),("Finland","Helsinki","Scandinavia"),
    ("Greece","Athens","Southern Europe"),("Portugal","Lisbon","Western Europe"),
    ("Austria","Vienna","Central Europe"),("Switzerland","Bern","Central Europe"),
    ("Czech Republic","Prague","Central Europe"),("Hungary","Budapest","Central Europe"),
    ("Romania","Bucharest","Eastern Europe"),("Ukraine","Kyiv","Eastern Europe"),
    ("Peru","Lima","South America"),("Colombia","Bogotá","South America"),
    ("Chile","Santiago","South America"),("Morocco","Rabat","North Africa"),
    ("Kenya","Nairobi","East Africa"),("Ethiopia","Addis Ababa","East Africa"),
    ("Iran","Tehran","the Middle East"),("Iraq","Baghdad","the Middle East"),
    ("Pakistan","Islamabad","South Asia"),("Indonesia","Jakarta","Southeast Asia"),
    ("Thailand","Bangkok","Southeast Asia"),("Vietnam","Hanoi","Southeast Asia"),
    ("Malaysia","Kuala Lumpur","Southeast Asia"),("Philippines","Manila","Southeast Asia"),
    ("Israel","Jerusalem","the Middle East"),("Ireland","Dublin","Western Europe"),
]

# person-achievement seeds
_PERSON_FACTS = [
    ("Alexander Graham Bell","invented the telephone","1876","communications"),
    ("Albert Einstein","developed the theory of relativity","1905","physics"),
    ("Charles Darwin","published the theory of evolution","1859","biology"),
    ("Isaac Newton","formulated the laws of motion and universal gravitation","1687","physics"),
    ("Marie Curie","discovered radium and polonium","1898","chemistry"),
    ("Thomas Edison","invented the phonograph and a practical light bulb","the 1870s","technology"),
    ("Nikola Tesla","pioneered alternating current electrical systems","the 1880s","engineering"),
    ("Tim Berners-Lee","invented the World Wide Web","1989","computer science"),
    ("Galileo Galilei","improved the telescope and confirmed heliocentrism","1610","astronomy"),
    ("William Shakespeare","wrote Romeo and Juliet and Hamlet","the late 1590s","literature"),
    ("J.K. Rowling","wrote the Harry Potter series","1997","literature"),
    ("Neil Armstrong","became the first human to walk on the Moon","1969","space exploration"),
    ("Leonardo da Vinci","painted the Mona Lisa","around 1503","art"),
    ("Charles Dickens","wrote Oliver Twist and Great Expectations","the 1830s–1860s","literature"),
    ("Pablo Picasso","co-founded Cubism","around 1907","art"),
    ("Mozart","composed over 600 works including Symphony No. 40","the 1780s","music"),
    ("Wright brothers","made the first powered airplane flight","1903","aviation"),
    ("James Watson","co-discovered the double-helix structure of DNA","1953","biology"),
    ("Nelson Mandela","became the first Black president of South Africa","1994","politics"),
    ("Mahatma Gandhi","led India's independence movement nonviolently","1947","politics"),
]

# landmark-location seeds
_LANDMARK_FACTS = [
    ("Eiffel Tower","France","Paris, on the Champ de Mars"),
    ("Great Wall of China","China","northern China, over 13,000 miles long"),
    ("Colosseum","Italy","Rome, in the heart of the city"),
    ("Taj Mahal","India","Agra, in the state of Uttar Pradesh"),
    ("Statue of Liberty","United States","New York Harbor"),
    ("Pyramids of Giza","Egypt","Giza plateau, near Cairo"),
    ("Big Ben","United Kingdom","London, at the Palace of Westminster"),
    ("Sydney Opera House","Australia","Sydney, on Bennelong Point"),
    ("Mount Everest","Nepal","the Himalayas on the Nepal–Tibet border"),
    ("Amazon River","Brazil","South America, flowing through the Amazon Basin"),
    ("Grand Canyon","United States","Arizona, carved by the Colorado River"),
    ("Great Barrier Reef","Australia","off the coast of Queensland"),
    ("Stonehenge","United Kingdom","Wiltshire, England"),
    ("Machu Picchu","Peru","the Andes Mountains"),
    ("Berlin Wall","Germany","Berlin, where it once divided the city"),
]

# country-language seeds
_COUNTRY_LANGUAGE = [
    ("France","French"),("Germany","German"),("Brazil","Portuguese"),
    ("Spain","Spanish"),("Japan","Japanese"),("China","Mandarin Chinese"),
    ("Russia","Russian"),("Italy","Italian"),("Netherlands","Dutch"),
    ("Greece","Greek"),("Poland","Polish"),("Sweden","Swedish"),
    ("Turkey","Turkish"),("Iran","Persian"),("Saudi Arabia","Arabic"),
    ("South Korea","Korean"),("Vietnam","Vietnamese"),("India","Hindi"),
    ("Ukraine","Ukrainian"),("Indonesia","Indonesian"),
]

# country-currency seeds
_COUNTRY_CURRENCY = [
    ("United States","US Dollar"),("United Kingdom","Pound Sterling"),
    ("Japan","Japanese Yen"),("Germany","Euro"),("China","Chinese Yuan"),
    ("Russia","Russian Ruble"),("India","Indian Rupee"),("Brazil","Brazilian Real"),
    ("Switzerland","Swiss Franc"),("South Korea","South Korean Won"),
]

# historical event seeds
_EVENT_FACTS = [
    ("World War II","1939–1945, ending with Allied victory over Germany and Japan"),
    ("World War I","1914–1918, ending with the Treaty of Versailles"),
    ("the French Revolution","1789–1799, transforming France from a monarchy into a republic"),
    ("the American Revolution","1775–1783, resulting in American independence from Britain"),
    ("the Moon landing","July 20 1969, when Neil Armstrong walked on the Moon"),
    ("the fall of the Berlin Wall","November 9 1989, reuniting East and West Germany"),
    ("the Industrial Revolution","the late 18th century, beginning in Great Britain"),
    ("the Great Depression","1929–1939, triggered by the Wall Street Crash"),
    ("the Korean War","1950–1953, ending in an armistice dividing the Korean Peninsula"),
    ("the Cold War","1947–1991, between the United States and the Soviet Union"),
]


def _build_seeds() -> list[tuple[str, str]]:
    """Return (question, correct_fact) tuples."""
    seeds = []

    for country, capital, region in _COUNTRY_CAPITAL:
        seeds.append((
            f"What is the capital of {country}?",
            f"The capital of {country} is {capital}, located in {region}.",
        ))
        seeds.append((
            f"Which city serves as the capital of {country}?",
            f"{capital} is the capital city of {country}.",
        ))

    for person, achievement, year, domain in _PERSON_FACTS:
        seeds.append((
            f"What is {person} known for?",
            f"{person} {achievement} in {year}, making a major contribution to {domain}.",
        ))
        seeds.append((
            f"Who {achievement}?",
            f"{person} {achievement} {year}.",
        ))

    for landmark, country, detail in _LANDMARK_FACTS:
        seeds.append((
            f"Where is the {landmark} located?",
            f"The {landmark} is located in {detail}, {country}.",
        ))

    for country, language in _COUNTRY_LANGUAGE:
        seeds.append((
            f"What language is spoken in {country}?",
            f"The official language of {country} is {language}.",
        ))

    for country, currency in _COUNTRY_CURRENCY:
        seeds.append((
            f"What is the currency of {country}?",
            f"The currency of {country} is the {currency}.",
        ))

    for event, date_context in _EVENT_FACTS:
        seeds.append((
            f"When did {event} occur?",
            f"{event.capitalize()} occurred {date_context}.",
        ))

    return seeds


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a dataset labeler generating training examples for a QA quality classifier. "
    "Return ONLY valid JSON, no markdown, no commentary."
)

def _make_prompt(question: str, correct_fact: str) -> str:
    return f"""\
Question: {question}
Correct fact: {correct_fact}

Generate exactly 15 answer examples spread across three quality tiers:

5 GOOD answers — complete, accurate, specific; naturally varying sentence structure and vocabulary; \
always name the key entity and give meaningful context.

3 ACCEPTABLE answers — correct but vague; deliberately omit the specific entity name or key detail; \
sound like a human who half-remembers the fact.

7 POOR answers — one of each subtype:
  (a) single word or two-word fragment only
  (b) completely off-topic fluent sentence (unrelated subject matter)
  (c) answer that pathologically repeats a key word or phrase 3+ times
  (d) words from the correct answer shuffled into wrong / broken order
  (e) response that is itself a question (evasive non-answer)
  (f) plausible-sounding but factually wrong entity
  (g) rambling filler that never states the actual fact

Return JSON:
{{"examples": [{{"answer": "...", "label": "good|acceptable|poor"}}]}}
Keep each answer under 60 words."""


# ── API call with retry ─────────────────────────────────────────────────────────

def _call_api(client: anthropic.Anthropic, question: str, correct_fact: str,
              max_retries: int = 3) -> list[dict] | None:
    prompt = _make_prompt(question, correct_fact)
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # strip optional markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            examples = parsed.get("examples", parsed) if isinstance(parsed, dict) else parsed
            # attach question + validate label
            result = []
            for ex in examples:
                label = ex.get("label", "").lower()
                if label not in ("good", "acceptable", "poor"):
                    continue
                answer = str(ex.get("answer", "")).strip()
                if answer:
                    result.append({"question": question, "answer": answer, "label": label})
            return result
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] parse failed after {max_retries} attempts: {e}")
                return None
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  [rate-limit] waiting {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"  [ERROR] {e}")
            return None
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N seeds (for smoke-testing).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip seeds already saved in the progress file.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    seeds = _build_seeds()
    print(f"Total seeds: {len(seeds)}")

    # Load already-processed questions if resuming
    done_questions: set[str] = set()
    prior_examples: list[dict] = []
    if args.resume and PROGRESS_PATH.exists():
        for line in PROGRESS_PATH.read_text().splitlines():
            ex = json.loads(line)
            prior_examples.append(ex)
            done_questions.add(ex["question"])
        print(f"Resuming: {len(done_questions)} seeds already done "
              f"({len(prior_examples)} examples loaded).")

    seeds_to_run = [s for s in seeds if s[0] not in done_questions]
    if args.limit:
        seeds_to_run = seeds_to_run[: args.limit]

    print(f"Seeds to process: {len(seeds_to_run)}")

    all_examples = list(prior_examples)
    progress_fh  = open(PROGRESS_PATH, "a")

    try:
        for idx, (question, correct_fact) in enumerate(seeds_to_run, 1):
            print(f"[{idx}/{len(seeds_to_run)}] {question[:70]}", end=" … ", flush=True)
            examples = _call_api(client, question, correct_fact)
            if examples is None:
                print("SKIP")
                continue
            for ex in examples:
                all_examples.append(ex)
                progress_fh.write(json.dumps(ex) + "\n")
            progress_fh.flush()
            dist = Counter(e["label"] for e in examples)
            print(f"got {len(examples)} ({dist['good']}g/{dist['acceptable']}a/{dist['poor']}p)")
            # polite pause to respect rate limits
            time.sleep(0.3)
    finally:
        progress_fh.close()

    # Save final output
    import random
    random.seed(42)
    random.shuffle(all_examples)
    with open(OUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    dist = Counter(e["label"] for e in all_examples)
    print(f"\nSaved {len(all_examples):,} examples → {OUT_PATH}")
    print(f"  good:       {dist['good']:,}  ({dist['good']/len(all_examples)*100:.1f}%)")
    print(f"  acceptable: {dist['acceptable']:,}  ({dist['acceptable']/len(all_examples)*100:.1f}%)")
    print(f"  poor:       {dist['poor']:,}  ({dist['poor']/len(all_examples)*100:.1f}%)")

    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
