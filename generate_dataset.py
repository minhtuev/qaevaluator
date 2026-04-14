"""
Generate ~5,000 labeled QA pairs from templates + a built-in knowledge base.
No LLM or external API required — all text is produced deterministically.

Run:    python generate_dataset.py
Output: data/qa_large.jsonl  {"question":..., "answer":..., "label":...}

Distribution target:  ~40% good  |  ~25% acceptable  |  ~35% poor
"""

import json
import random
from pathlib import Path

random.seed(42)
Path("data").mkdir(exist_ok=True)
OUT = Path("data/qa_large.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge base
# ══════════════════════════════════════════════════════════════════════════════

# (country, capital, region)
COUNTRY_CAPITAL = [
    ("Afghanistan", "Kabul", "Central Asia"),
    ("Argentina", "Buenos Aires", "South America"),
    ("Australia", "Canberra", "Oceania"),
    ("Austria", "Vienna", "Central Europe"),
    ("Belgium", "Brussels", "Western Europe"),
    ("Brazil", "Brasília", "South America"),
    ("Canada", "Ottawa", "North America"),
    ("Chile", "Santiago", "South America"),
    ("China", "Beijing", "East Asia"),
    ("Colombia", "Bogotá", "South America"),
    ("Cuba", "Havana", "the Caribbean"),
    ("Czech Republic", "Prague", "Central Europe"),
    ("Denmark", "Copenhagen", "Scandinavia"),
    ("Egypt", "Cairo", "North Africa"),
    ("Ethiopia", "Addis Ababa", "East Africa"),
    ("Finland", "Helsinki", "Scandinavia"),
    ("France", "Paris", "Western Europe"),
    ("Germany", "Berlin", "Central Europe"),
    ("Ghana", "Accra", "West Africa"),
    ("Greece", "Athens", "Southern Europe"),
    ("Hungary", "Budapest", "Central Europe"),
    ("India", "New Delhi", "South Asia"),
    ("Indonesia", "Jakarta", "Southeast Asia"),
    ("Iran", "Tehran", "the Middle East"),
    ("Iraq", "Baghdad", "the Middle East"),
    ("Ireland", "Dublin", "Western Europe"),
    ("Israel", "Jerusalem", "the Middle East"),
    ("Italy", "Rome", "Southern Europe"),
    ("Japan", "Tokyo", "East Asia"),
    ("Jordan", "Amman", "the Middle East"),
    ("Kenya", "Nairobi", "East Africa"),
    ("Malaysia", "Kuala Lumpur", "Southeast Asia"),
    ("Mexico", "Mexico City", "North America"),
    ("Morocco", "Rabat", "North Africa"),
    ("Nepal", "Kathmandu", "South Asia"),
    ("Netherlands", "Amsterdam", "Western Europe"),
    ("New Zealand", "Wellington", "Oceania"),
    ("Nigeria", "Abuja", "West Africa"),
    ("Norway", "Oslo", "Scandinavia"),
    ("Pakistan", "Islamabad", "South Asia"),
    ("Peru", "Lima", "South America"),
    ("Philippines", "Manila", "Southeast Asia"),
    ("Poland", "Warsaw", "Central Europe"),
    ("Portugal", "Lisbon", "Western Europe"),
    ("Romania", "Bucharest", "Eastern Europe"),
    ("Russia", "Moscow", "Eastern Europe"),
    ("Saudi Arabia", "Riyadh", "the Middle East"),
    ("South Africa", "Pretoria", "Southern Africa"),
    ("South Korea", "Seoul", "East Asia"),
    ("Spain", "Madrid", "Southern Europe"),
    ("Sri Lanka", "Colombo", "South Asia"),
    ("Sweden", "Stockholm", "Scandinavia"),
    ("Switzerland", "Bern", "Central Europe"),
    ("Thailand", "Bangkok", "Southeast Asia"),
    ("Turkey", "Ankara", "Western Asia"),
    ("Ukraine", "Kyiv", "Eastern Europe"),
    ("United Kingdom", "London", "Western Europe"),
    ("United States", "Washington", "North America"),
    ("Venezuela", "Caracas", "South America"),
    ("Vietnam", "Hanoi", "Southeast Asia"),
]

# (person, achievement_phrase, year_phrase, domain)
PERSON_FACTS = [
    ("Alexander Graham Bell",  "invented the telephone",                          "in 1876",         "communications"),
    ("Albert Einstein",        "developed the theory of relativity",               "in 1905",         "physics"),
    ("Charles Darwin",         "published the theory of evolution by natural selection", "in 1859",   "biology"),
    ("Isaac Newton",           "formulated the laws of motion and universal gravitation", "in 1687",  "physics"),
    ("Marie Curie",            "discovered radium and polonium",                   "in 1898",         "chemistry"),
    ("Thomas Edison",          "invented the phonograph and a practical light bulb", "in the 1870s", "technology"),
    ("Nikola Tesla",           "pioneered alternating current electrical systems", "in the 1880s",    "electrical engineering"),
    ("Tim Berners-Lee",        "invented the World Wide Web",                      "in 1989",         "computer science"),
    ("Galileo Galilei",        "improved the telescope and confirmed heliocentrism", "in 1610",       "astronomy"),
    ("Copernicus",             "proposed the heliocentric model of the solar system", "in 1543",      "astronomy"),
    ("Leonardo da Vinci",      "painted the Mona Lisa",                            "around 1503",     "art"),
    ("William Shakespeare",    "wrote Romeo and Juliet",                            "around 1597",     "literature"),
    ("J.K. Rowling",           "wrote the Harry Potter series",                    "starting in 1997","literature"),
    ("Stephen Hawking",        "made groundbreaking contributions to black hole theory", "in the 1970s", "physics"),
    ("Neil Armstrong",         "became the first person to walk on the Moon",      "in 1969",         "space exploration"),
    ("Christopher Columbus",   "completed the first European voyage to the Americas", "in 1492",      "exploration"),
    ("Mahatma Gandhi",         "led India's independence movement through nonviolent resistance", "in 1947", "politics"),
    ("Nelson Mandela",         "became the first Black president of South Africa", "in 1994",         "politics"),
    ("Florence Nightingale",   "founded modern nursing practices",                 "in the 1850s",    "medicine"),
    ("Sigmund Freud",          "founded psychoanalysis",                           "in 1899",         "psychology"),
    ("Charles Dickens",        "wrote Oliver Twist and A Tale of Two Cities",      "in the 1830s",    "literature"),
    ("Pablo Picasso",          "co-founded the Cubist movement",                   "around 1907",     "art"),
    ("Beethoven",              "composed the Ninth Symphony",                      "in 1824",         "music"),
    ("Mozart",                 "composed The Magic Flute",                         "in 1791",         "music"),
    ("Wright brothers",        "made the first successful powered airplane flight","in 1903",         "aviation"),
    ("James Watson",           "co-discovered the double-helix structure of DNA",  "in 1953",         "biology"),
    ("Pythagoras",             "developed the Pythagorean theorem",                "around 500 BC",   "mathematics"),
    ("Alan Turing",            "laid the theoretical foundations of computer science", "in 1936",     "computing"),
    ("Rembrandt",              "painted The Night Watch",                          "in 1642",         "art"),
    ("Vincent van Gogh",       "painted The Starry Night",                         "in 1889",         "art"),
]

# (landmark, country, location_detail)
LANDMARK_FACTS = [
    ("Eiffel Tower",         "France",          "Paris, on the Champ de Mars"),
    ("Great Wall of China",  "China",           "northern China, stretching over 13,000 miles"),
    ("Colosseum",            "Italy",           "Rome, in the heart of the city"),
    ("Taj Mahal",            "India",           "Agra, in the state of Uttar Pradesh"),
    ("Statue of Liberty",    "United States",   "New York Harbor"),
    ("Pyramids of Giza",     "Egypt",           "Giza, near Cairo"),
    ("Big Ben",              "United Kingdom",  "London, at the north end of the Palace of Westminster"),
    ("Sydney Opera House",   "Australia",       "Sydney, on Bennelong Point"),
    ("Stonehenge",           "United Kingdom",  "Wiltshire, England"),
    ("Burj Khalifa",         "United Arab Emirates", "Dubai"),
    ("Golden Gate Bridge",   "United States",   "San Francisco Bay"),
    ("Parthenon",            "Greece",          "Athens, atop the Acropolis"),
    ("Mount Everest",        "Nepal",           "the Himalayas on the Nepal–Tibet border"),
    ("Amazon River",         "Brazil",          "South America, flowing through the Amazon Basin"),
    ("Niagara Falls",        "Canada",          "the border between Ontario, Canada and New York, USA"),
    ("Grand Canyon",         "United States",   "Arizona, carved by the Colorado River"),
    ("Great Barrier Reef",   "Australia",       "the Coral Sea, off the coast of Queensland"),
    ("Victoria Falls",       "Zimbabwe",        "the border of Zimbabwe and Zambia"),
    ("Leaning Tower of Pisa","Italy",           "Pisa, in the Tuscany region"),
    ("Berlin Wall",          "Germany",         "Berlin, where it once divided the city"),
    ("Acropolis",            "Greece",          "Athens, overlooking the city"),
    ("Angkor Wat",           "Cambodia",        "Siem Reap Province"),
    ("Machu Picchu",         "Peru",            "the Andes Mountains"),
    ("Sahara Desert",        "Algeria",         "North Africa, spanning multiple countries"),
    ("Pacific Ocean",        "none",            "between Asia/Australia and the Americas"),
]

# (country, language)
COUNTRY_LANGUAGE = [
    ("France",        "French"),
    ("Germany",       "German"),
    ("Brazil",        "Portuguese"),
    ("Portugal",      "Portuguese"),
    ("Spain",         "Spanish"),
    ("Mexico",        "Spanish"),
    ("Japan",         "Japanese"),
    ("China",         "Mandarin Chinese"),
    ("Russia",        "Russian"),
    ("Italy",         "Italian"),
    ("Netherlands",   "Dutch"),
    ("Greece",        "Greek"),
    ("Poland",        "Polish"),
    ("Sweden",        "Swedish"),
    ("Norway",        "Norwegian"),
    ("Denmark",       "Danish"),
    ("Finland",       "Finnish"),
    ("Turkey",        "Turkish"),
    ("Iran",          "Persian"),
    ("Saudi Arabia",  "Arabic"),
    ("Egypt",         "Arabic"),
    ("Israel",        "Hebrew"),
    ("South Korea",   "Korean"),
    ("Vietnam",       "Vietnamese"),
    ("Thailand",      "Thai"),
    ("Indonesia",     "Indonesian"),
    ("India",         "Hindi"),
    ("Pakistan",      "Urdu"),
    ("Ukraine",       "Ukrainian"),
    ("Romania",       "Romanian"),
]

# (country, currency, symbol_note)
COUNTRY_CURRENCY = [
    ("United States",  "US Dollar",         "the world's primary reserve currency"),
    ("United Kingdom", "Pound Sterling",    "one of the oldest currencies still in use"),
    ("Japan",          "Japanese Yen",      "one of the most traded currencies globally"),
    ("Germany",        "Euro",              "shared by most European Union members"),
    ("France",         "Euro",              "adopted in 1999"),
    ("China",          "Chinese Yuan",      "also known as the Renminbi"),
    ("Russia",         "Russian Ruble",     "one of the oldest currencies in the world"),
    ("India",          "Indian Rupee",      "subdivided into 100 paise"),
    ("Brazil",         "Brazilian Real",    "introduced in 1994"),
    ("Canada",         "Canadian Dollar",   "also called the loonie"),
    ("Australia",      "Australian Dollar", "introduced in 1966"),
    ("Switzerland",    "Swiss Franc",       "known for its stability"),
    ("Mexico",         "Mexican Peso",      "one of the most traded currencies in Latin America"),
    ("South Korea",    "South Korean Won",  "subdivided into 100 jeon"),
    ("Sweden",         "Swedish Krona",     "one of the major Scandinavian currencies"),
]

# (event, date_phrase, context_phrase)
EVENT_FACTS = [
    ("World War II",          "from 1939 to 1945",    "ending with the Allied victory over Germany and Japan"),
    ("World War I",           "from 1914 to 1918",    "ending with the Treaty of Versailles"),
    ("The French Revolution", "from 1789 to 1799",    "transforming France from a monarchy into a republic"),
    ("The American Revolution","from 1775 to 1783",   "resulting in American independence from Britain"),
    ("The Cold War",          "from 1947 to 1991",    "between the United States and the Soviet Union"),
    ("The Renaissance",       "from the 14th to 17th centuries", "beginning in Italy and spreading across Europe"),
    ("The Industrial Revolution","in the late 18th century","beginning in Great Britain"),
    ("The Moon landing",      "on July 20, 1969",     "when Neil Armstrong became the first person to walk on the Moon"),
    ("The fall of the Berlin Wall", "on November 9, 1989", "reuniting East and West Germany"),
    ("The Black Death",       "from 1347 to 1351",    "killing an estimated one-third of Europe's population"),
    ("The Great Depression",  "from 1929 to 1939",    "triggered by the Wall Street Crash of 1929"),
    ("The Vietnam War",       "from 1955 to 1975",    "ending with North Vietnamese victory and reunification"),
    ("The Korean War",        "from 1950 to 1953",    "ending in an armistice that divided the Korean Peninsula"),
    ("The September 11 attacks", "on September 11, 2001", "targeting New York City and Washington DC"),
    ("The Space Race",        "from the late 1950s to 1969", "between the United States and the Soviet Union"),
]

# Off-topic fluent sentences (pool for poor answers)
OFF_TOPIC = [
    "The weather today is sunny and warm with a gentle breeze.",
    "Pizza is a popular Italian dish enjoyed around the world.",
    "My favorite hobby is photography and I take pictures every weekend.",
    "The stock market experienced significant fluctuations yesterday.",
    "I really enjoy listening to classical music in the evenings.",
    "Cats and dogs are the most popular household pets worldwide.",
    "The best way to make pasta is to cook it in salted boiling water.",
    "Chocolate comes in many varieties, including milk and dark.",
    "The new smartphone model features an improved camera system.",
    "Gardening is a relaxing hobby that many people enjoy on weekends.",
    "Coffee is one of the most widely consumed beverages globally.",
    "The library contains thousands of books on diverse subjects.",
    "Swimming is excellent exercise and is very easy on the joints.",
    "Autumn is a beautiful season with colorful falling leaves.",
    "The train departs from platform seven at half past noon.",
    "Reading before bedtime is a great habit for relaxation.",
    "Traffic was particularly heavy during the morning rush hour.",
    "The restaurant downtown is known for its excellent seafood dishes.",
    "Hiking in the mountains is a wonderful way to stay active.",
    "The new film received excellent reviews from professional critics.",
    "Cooking at home is healthier and more economical than eating out.",
    "Online shopping has become increasingly popular over recent years.",
    "The museum recently opened a new wing dedicated to modern art.",
    "Cycling to work is an eco-friendly and healthy commuting option.",
    "The annual music festival attracts thousands of visitors each year.",
]

# Evasive question responses (poor answers)
QUESTION_RESPS = [
    "Why do you want to know?",
    "Isn't that something you could look up yourself?",
    "Have you tried reading a textbook about this?",
    "I'm not sure, what do you think?",
    "Does it really matter?",
    "Why is that relevant to you?",
    "Couldn't you find that out yourself?",
    "That is a very interesting question, isn't it?",
    "I wonder why you are asking me this.",
    "Maybe you should consult an expert.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Template helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pick(seq):
    return random.choice(seq)

def _scramble(sentence: str) -> str:
    words = sentence.rstrip(".!?").split()
    random.shuffle(words)
    return " ".join(words).capitalize() + "."

def _repeat_phrase(phrase: str, n: int = 3) -> str:
    parts = " ".join([phrase] * n)
    return f"{parts} {parts}."

def _wrong_entity(correct: str, pool: list[str]) -> str:
    alts = [e for e in pool if e.lower() != correct.lower()]
    return _pick(alts) + "." if alts else "Unknown."


# ══════════════════════════════════════════════════════════════════════════════
# Per-category example builders
# ══════════════════════════════════════════════════════════════════════════════

def _capital_examples(country, capital, region, all_capitals):
    questions = [
        f"What is the capital of {country}?",
        f"Which city serves as the capital of {country}?",
        f"What is the capital city of {country}?",
    ]
    good_tmpls = [
        f"{capital} is the capital city of {country}.",
        f"The capital of {country} is {capital}, located in {region}.",
        f"{capital} serves as the capital and main political center of {country}.",
        f"{country}'s capital city is {capital}, situated in {region}.",
    ]
    acc_tmpls = [
        f"The capital is {capital}.",
        f"I believe the capital is {capital}.",
        f"It is {capital}, a major city in {country}.",
        f"The answer would be {capital}.",
    ]
    examples = []
    for q in questions:
        for t in random.sample(good_tmpls, 3):
            examples.append({"question": q, "answer": t, "label": "good"})
        for t in random.sample(acc_tmpls, 2):
            examples.append({"question": q, "answer": t, "label": "acceptable"})
        examples += [
            {"question": q, "answer": f"{capital}.",                         "label": "poor"},
            {"question": q, "answer": _wrong_entity(capital, all_capitals),  "label": "poor"},
            {"question": q, "answer": _repeat_phrase(capital),               "label": "poor"},
            {"question": q, "answer": _scramble(f"the capital of {country} is {capital}"), "label": "poor"},
            {"question": q, "answer": _pick(OFF_TOPIC),                      "label": "poor"},
        ]
    return examples


def _person_examples(person, achievement, year, domain, all_persons):
    questions = [
        f"Who {achievement.split()[0]}ed the {' '.join(achievement.split()[1:])}?" if achievement.split()[0] in ("invent","discover","found","develop","pioneer","publish","propose","compose","paint","write") else f"What is {person} known for?",
        f"What is {person} famous for?",
        f"Which scientist or thinker is associated with {' '.join(achievement.split()[1:][:3])}?",
    ]
    # simplify: just use three generic question types
    q_templates = [
        f"What is {person} known for?",
        f"Who {achievement}?",
        f"What major contribution did {person} make to {domain}?",
    ]
    good_tmpls = [
        f"{person} is known for having {achievement} {year}.",
        f"{person} {achievement} {year}, making a landmark contribution to {domain}.",
        f"The famous {domain} figure {person} {achievement} {year}.",
        f"{person} is celebrated for having {achievement}, a major milestone in {domain}.",
    ]
    acc_tmpls = [
        f"{person} made important contributions to {domain}.",
        f"They were a famous {domain} figure who lived centuries ago.",
        f"Someone known for work in {domain} did this.",
        f"{person} is a well-known historical figure in {domain}.",
    ]
    examples = []
    for q in q_templates:
        for t in random.sample(good_tmpls, 3):
            examples.append({"question": q, "answer": t, "label": "good"})
        for t in random.sample(acc_tmpls, 2):
            examples.append({"question": q, "answer": t, "label": "acceptable"})
        examples += [
            {"question": q, "answer": f"{person}.",                        "label": "poor"},
            {"question": q, "answer": _wrong_entity(person, all_persons),  "label": "poor"},
            {"question": q, "answer": _repeat_phrase(person),              "label": "poor"},
            {"question": q, "answer": _scramble(f"{person} {achievement} {year}"), "label": "poor"},
            {"question": q, "answer": _pick(OFF_TOPIC),                    "label": "poor"},
        ]
    return examples


def _landmark_examples(landmark, country, detail, all_landmarks):
    q_templates = [
        f"Where is the {landmark} located?",
        f"In which country can you find the {landmark}?",
        f"What country is the {landmark} in?",
    ]
    good_tmpls = [
        f"The {landmark} is located in {detail}.",
        f"The {landmark} can be found in {country}, specifically in {detail}.",
        f"Located in {detail}, the {landmark} is one of the most famous sites in {country}.",
        f"The {landmark} is situated in {country}, in {detail}.",
    ]
    acc_tmpls = [
        f"It is located in {country}.",
        f"The {landmark} is in {country}, I believe.",
        f"I think it is somewhere in {country}.",
        f"It is a famous landmark found in {country}.",
    ]
    examples = []
    for q in q_templates:
        for t in random.sample(good_tmpls, 3):
            examples.append({"question": q, "answer": t, "label": "good"})
        for t in random.sample(acc_tmpls, 2):
            examples.append({"question": q, "answer": t, "label": "acceptable"})
        examples += [
            {"question": q, "answer": f"{country}.",                                "label": "poor"},
            {"question": q, "answer": _wrong_entity(country, [l for _, l, _ in LANDMARK_FACTS]), "label": "poor"},
            {"question": q, "answer": _repeat_phrase(landmark),                     "label": "poor"},
            {"question": q, "answer": _scramble(f"the {landmark} is located in {detail}"), "label": "poor"},
            {"question": q, "answer": _pick(OFF_TOPIC),                             "label": "poor"},
        ]
    return examples


def _language_examples(country, language, all_languages):
    q_templates = [
        f"What language is spoken in {country}?",
        f"What is the official language of {country}?",
        f"Which language do people speak in {country}?",
    ]
    good_tmpls = [
        f"The official language of {country} is {language}.",
        f"{language} is the primary language spoken in {country}.",
        f"In {country}, the population speaks {language} as the official language.",
        f"{language} is the national and official language of {country}.",
    ]
    acc_tmpls = [
        f"The language is {language}.",
        f"They speak {language} in {country}.",
        f"I believe {language} is the main language.",
        f"People there generally speak {language}.",
    ]
    examples = []
    for q in q_templates:
        for t in random.sample(good_tmpls, 3):
            examples.append({"question": q, "answer": t, "label": "good"})
        for t in random.sample(acc_tmpls, 2):
            examples.append({"question": q, "answer": t, "label": "acceptable"})
        examples += [
            {"question": q, "answer": f"{language}.",                           "label": "poor"},
            {"question": q, "answer": _wrong_entity(language, all_languages),   "label": "poor"},
            {"question": q, "answer": _repeat_phrase(language),                 "label": "poor"},
            {"question": q, "answer": _scramble(f"the official language of {country} is {language}"), "label": "poor"},
            {"question": q, "answer": _pick(OFF_TOPIC),                         "label": "poor"},
        ]
    return examples


def _currency_examples(country, currency, note, all_currencies):
    q_templates = [
        f"What is the currency of {country}?",
        f"What currency does {country} use?",
        f"Which currency is used in {country}?",
    ]
    good_tmpls = [
        f"The currency of {country} is the {currency}, {note}.",
        f"{country} uses the {currency} as its official currency.",
        f"The official currency of {country} is the {currency}.",
        f"In {country}, the {currency} is the national currency, {note}.",
    ]
    acc_tmpls = [
        f"The currency is the {currency}.",
        f"They use the {currency}.",
        f"I believe the currency is the {currency}.",
        f"{country} has its own currency called the {currency}.",
    ]
    examples = []
    for q in q_templates:
        for t in random.sample(good_tmpls, 3):
            examples.append({"question": q, "answer": t, "label": "good"})
        for t in random.sample(acc_tmpls, 2):
            examples.append({"question": q, "answer": t, "label": "acceptable"})
        examples += [
            {"question": q, "answer": f"{currency}.",                           "label": "poor"},
            {"question": q, "answer": _wrong_entity(currency, all_currencies),  "label": "poor"},
            {"question": q, "answer": _repeat_phrase(currency),                 "label": "poor"},
            {"question": q, "answer": _scramble(f"the currency of {country} is the {currency}"), "label": "poor"},
            {"question": q, "answer": _pick(OFF_TOPIC),                         "label": "poor"},
        ]
    return examples


def _event_examples(event, date_phrase, context):
    q_templates = [
        f"When did {event} occur?",
        f"When did {event} take place?",
        f"What were the dates of {event}?",
    ]
    good_tmpls = [
        f"{event} occurred {date_phrase}, {context}.",
        f"{event} took place {date_phrase}, {context}.",
        f"Historians date {event} {date_phrase}; it is remembered for {context}.",
        f"{event} happened {date_phrase}, a pivotal period marked by {context}.",
    ]
    acc_tmpls = [
        f"It happened {date_phrase}.",
        f"{event} took place {date_phrase}.",
        f"I believe it occurred sometime around that period.",
        f"It was during a significant historical period.",
    ]
    examples = []
    for q in q_templates:
        for t in random.sample(good_tmpls, 3):
            examples.append({"question": q, "answer": t, "label": "good"})
        for t in random.sample(acc_tmpls, 2):
            examples.append({"question": q, "answer": t, "label": "acceptable"})
        examples += [
            {"question": q, "answer": date_phrase + ".",                         "label": "poor"},
            {"question": q, "answer": _pick(OFF_TOPIC),                          "label": "poor"},
            {"question": q, "answer": _repeat_phrase(event.split()[0]),           "label": "poor"},
            {"question": q, "answer": _scramble(f"{event} occurred {date_phrase}"), "label": "poor"},
            {"question": q, "answer": _pick(QUESTION_RESPS),                     "label": "poor"},
        ]
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# Main generator
# ══════════════════════════════════════════════════════════════════════════════

def generate() -> list[dict]:
    all_capitals  = [cap for _, cap, _ in COUNTRY_CAPITAL]
    all_persons   = [p   for p, _, _, _ in PERSON_FACTS]
    all_languages = [lng for _, lng in COUNTRY_LANGUAGE]
    all_currencies= [cur for _, cur, _ in COUNTRY_CURRENCY]

    examples: list[dict] = []

    for country, capital, region in COUNTRY_CAPITAL:
        examples.extend(_capital_examples(country, capital, region, all_capitals))

    for person, achievement, year, domain in PERSON_FACTS:
        examples.extend(_person_examples(person, achievement, year, domain, all_persons))

    for landmark, country, detail in LANDMARK_FACTS:
        examples.extend(_landmark_examples(landmark, country, detail,
                                           [lm for lm, _, _ in LANDMARK_FACTS]))

    for country, language in COUNTRY_LANGUAGE:
        examples.extend(_language_examples(country, language, all_languages))

    for country, currency, note in COUNTRY_CURRENCY:
        examples.extend(_currency_examples(country, currency, note, all_currencies))

    for event, date_phrase, context in EVENT_FACTS:
        examples.extend(_event_examples(event, date_phrase, context))

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    data = generate()

    from collections import Counter
    dist = Counter(d["label"] for d in data)
    print(f"Generated {len(data):,} examples")
    print(f"  good:       {dist['good']:,}  ({dist['good']/len(data)*100:.1f}%)")
    print(f"  acceptable: {dist['acceptable']:,}  ({dist['acceptable']/len(data)*100:.1f}%)")
    print(f"  poor:       {dist['poor']:,}  ({dist['poor']/len(data)*100:.1f}%)")

    with open(OUT, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved → {OUT}")
