"""
Gazetteer of common named entities organized by category.

Each entry maps a lowercase phrase to its category.  Matching is
whole-phrase / whole-word so "bell" only fires when the answer
contains the standalone word "bell", not "bellboy".
"""

from typing import Optional
import re

# ── Entity data ───────────────────────────────────────────────────────────────

GAZETTEER: dict[str, str] = {}

_RAW: list[tuple[str, list[str]]] = [

    # Countries
    ("COUNTRY", [
        "Afghanistan", "Albania", "Algeria", "Argentina", "Australia",
        "Austria", "Bangladesh", "Belgium", "Bolivia", "Brazil",
        "Cambodia", "Canada", "Chile", "China", "Colombia",
        "Cuba", "Czech Republic", "Denmark", "Ecuador", "Egypt",
        "Ethiopia", "Finland", "France", "Germany", "Ghana",
        "Greece", "Guatemala", "Hungary", "India", "Indonesia",
        "Iran", "Iraq", "Ireland", "Israel", "Italy",
        "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
        "Lebanon", "Libya", "Malaysia", "Mexico", "Morocco",
        "Nepal", "Netherlands", "New Zealand", "Nigeria", "North Korea",
        "Norway", "Pakistan", "Peru", "Philippines", "Poland",
        "Portugal", "Romania", "Russia", "Saudi Arabia", "Serbia",
        "Singapore", "Somalia", "South Africa", "South Korea", "Spain",
        "Sri Lanka", "Sudan", "Sweden", "Switzerland", "Syria",
        "Taiwan", "Thailand", "Turkey", "Ukraine", "United Kingdom",
        "United States", "Uruguay", "Venezuela", "Vietnam", "Zimbabwe",
    ]),

    # Capital cities
    ("CAPITAL", [
        "Abu Dhabi", "Accra", "Addis Ababa", "Amman", "Amsterdam",
        "Ankara", "Athens", "Baghdad", "Bangkok", "Beijing",
        "Beirut", "Belgrade", "Berlin", "Bogotá", "Brasília",
        "Brussels", "Bucharest", "Budapest", "Buenos Aires", "Cairo",
        "Canberra", "Caracas", "Copenhagen", "Dhaka", "Dublin",
        "Hanoi", "Havana", "Helsinki", "Jakarta", "Kabul",
        "Kampala", "Kathmandu", "Khartoum", "Kiev", "Kyiv",
        "Kuala Lumpur", "Lagos", "Lima", "Lisbon", "Ljubljana",
        "London", "Luanda", "Luxembourg", "Madrid", "Manila",
        "Minsk", "Mogadishu", "Moscow", "Nairobi", "New Delhi",
        "Niamey", "Oslo", "Ottawa", "Paris", "Prague",
        "Pretoria", "Pyongyang", "Quito", "Rabat", "Riyadh",
        "Rome", "Santiago", "Seoul", "Singapore", "Sofia",
        "Stockholm", "Taipei", "Tashkent", "Tehran", "Tokyo",
        "Tripoli", "Tunis", "Vienna", "Warsaw", "Washington",
        "Wellington", "Yangon", "Yerevan", "Zagreb",
    ]),

    # Cities (non-capital but prominent)
    ("CITY", [
        "Barcelona", "Chicago", "Dubai", "Hong Kong", "Houston",
        "Istanbul", "Los Angeles", "Melbourne", "Miami", "Milan",
        "Mumbai", "Munich", "New York", "New York City", "Osaka",
        "São Paulo", "Seattle", "Shanghai", "Sydney", "Toronto",
        "Vancouver",
    ]),

    # Historical and contemporary persons
    ("PERSON", [
        "Abraham Lincoln", "Adele", "Albert Einstein", "Alexander Graham Bell",
        "Alexander the Great", "Amelia Earhart", "Aristotle",
        "Barack Obama", "Beethoven", "Bill Gates", "Charles Darwin",
        "Charles Dickens", "Christopher Columbus", "Cleopatra",
        "Confucius", "Copernicus", "Dalai Lama", "David Bowie",
        "Donald Trump", "Elon Musk", "Florence Nightingale",
        "Frida Kahlo", "Galileo Galilei", "Genghis Khan", "George Washington",
        "Harriet Tubman", "Helen Keller", "Isaac Newton", "J.K. Rowling",
        "James Watson", "Joan of Arc", "Julius Caesar", "Kepler",
        "Leonardo da Vinci", "Mahatma Gandhi", "Marie Curie",
        "Mark Zuckerberg", "Martin Luther King", "Michael Jordan",
        "Michelangelo", "Mozart", "Napoleon Bonaparte", "Neil Armstrong",
        "Nelson Mandela", "Nikola Tesla", "Pablo Picasso",
        "Plato", "Pythagoras", "Rembrandt", "Richard Feynman",
        "Rosa Parks", "Shakespeare", "Sigmund Freud", "Socrates",
        "Stephen Hawking", "Steve Jobs", "Tim Berners-Lee",
        "Thomas Edison", "Thomas Jefferson", "Vincent van Gogh",
        "Voltaire", "Walt Disney", "William Shakespeare", "Winston Churchill",
        "Wright brothers",
    ]),

    # Landmarks and natural features
    ("LANDMARK", [
        "Amazon River", "Angel Falls", "Arctic Ocean", "Atlantic Ocean",
        "Berlin Wall", "Big Ben", "Burj Khalifa", "Caribbean Sea",
        "Colosseum", "Dead Sea", "Eiffel Tower", "Empire State Building",
        "Galápagos Islands", "Golden Gate Bridge", "Grand Canyon",
        "Great Barrier Reef", "Great Wall of China", "Himalayas",
        "Indian Ocean", "Kilimanjaro", "Leaning Tower of Pisa",
        "Mediterranean Sea", "Mississippi River", "Mount Everest",
        "Niagara Falls", "Nile River", "Pacific Ocean", "Panama Canal",
        "Parthenon", "Pyramids of Giza", "Red Sea", "Rhine River",
        "Rocky Mountains", "Sahara Desert", "Statue of Liberty",
        "Stonehenge", "Suez Canal", "Sydney Opera House",
        "Taj Mahal", "Thames River", "Vatican", "Victoria Falls",
    ]),

    # Historical events
    ("EVENT", [
        "American Revolution", "Arab Spring", "Battle of Waterloo",
        "Black Death", "Cold War", "French Revolution",
        "Great Depression", "Holocaust", "Industrial Revolution",
        "Korean War", "Manhattan Project", "Moon landing",
        "Protestant Reformation", "Renaissance", "Roman Empire",
        "September 11", "Space Race", "Vietnam War",
        "World War I", "World War II",
    ]),

    # Scientific concepts and terms
    ("SCIENCE", [
        "Big Bang", "Black hole", "DNA", "Higgs boson",
        "Milky Way", "Periodic Table", "Pythagorean theorem",
        "Theory of Relativity", "Tectonic plates",
    ]),

    # Works of art, literature, and media
    ("WORK", [
        "1984", "Anna Karenina", "Crime and Punishment",
        "Don Quixote", "Gone with the Wind", "Harry Potter",
        "Les Misérables", "Macbeth", "Moby Dick", "Mona Lisa",
        "Odyssey", "Pride and Prejudice", "Romeo and Juliet",
        "Sistine Chapel", "The Great Gatsby", "The Iliad",
        "The Lord of the Rings", "Starry Night",
        "Declaration of Independence", "Magna Carta",
    ]),

    # Brands and organizations
    ("BRAND", [
        "Airbus", "Amazon", "Apple", "BMW", "Boeing",
        "Coca-Cola", "Ferrari", "Ford", "Google", "IBM",
        "Intel", "McDonald's", "Mercedes-Benz", "Meta", "Microsoft",
        "NASA", "Netflix", "Nike", "Nvidia", "OpenAI",
        "Samsung", "Sony", "SpaceX", "Tesla", "Toyota",
        "Twitter", "United Nations", "Volkswagen", "WHO", "Walmart",
    ]),

    # Languages
    ("LANGUAGE", [
        "Arabic", "Bengali", "Chinese", "Dutch", "English",
        "French", "German", "Greek", "Hebrew", "Hindi",
        "Indonesian", "Italian", "Japanese", "Korean", "Latin",
        "Malay", "Mandarin", "Persian", "Polish", "Portuguese",
        "Punjabi", "Russian", "Spanish", "Swahili", "Swedish",
        "Tamil", "Turkish", "Ukrainian", "Urdu", "Vietnamese",
    ]),

    # Currencies
    ("CURRENCY", [
        "Bitcoin", "Dollar", "Euro", "Franc", "Krona",
        "Peso", "Pound", "Real", "Renminbi", "Ruble",
        "Rupee", "Won", "Yen", "Yuan",
    ]),

    # Planets and celestial bodies
    ("CELESTIAL", [
        "Earth", "Jupiter", "Mars", "Mercury", "Moon",
        "Neptune", "Pluto", "Saturn", "Sun", "Uranus",
        "Venus", "Milky Way", "Andromeda",
    ]),

    # Continents
    ("CONTINENT", [
        "Africa", "Antarctica", "Asia", "Australia",
        "Europe", "North America", "South America",
    ]),

    # Religions
    ("RELIGION", [
        "Buddhism", "Christianity", "Hinduism", "Islam",
        "Judaism", "Sikhism",
    ]),
]

# Build the lookup table (lowercase phrase → category)
for _cat, _entries in _RAW:
    for _entry in _entries:
        GAZETTEER[_entry.lower()] = _cat

# Pre-sort phrases longest-first so multi-word phrases match before substrings
_SORTED_PHRASES: list[tuple[str, str]] = sorted(
    GAZETTEER.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ── Matching ──────────────────────────────────────────────────────────────────

def _word_boundary_pattern(phrase: str) -> re.Pattern:
    """Build a case-insensitive whole-phrase regex."""
    escaped = re.escape(phrase)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


_PATTERN_CACHE: dict[str, re.Pattern] = {}


def find_matches(text: str) -> list[tuple[str, str]]:
    """
    Return a list of (phrase, category) for every gazetteer entry that
    appears as a complete word / phrase in *text*.
    Longer phrases are tried first so 'Great Wall of China' wins over 'China'.
    """
    found: list[tuple[str, str]] = []
    remaining = text
    for phrase, category in _SORTED_PHRASES:
        if phrase not in _PATTERN_CACHE:
            _PATTERN_CACHE[phrase] = _word_boundary_pattern(phrase)
        if _PATTERN_CACHE[phrase].search(remaining):
            found.append((phrase, category))
    return found


def gazetteer_score(question: str, answer: str) -> float:
    """
    Score based on whether the answer contains a recognized named entity.

    Scoring logic
    -------------
    - If the answer contains a gazetteer entry whose category aligns with
      the question type (e.g. CAPITAL for 'where is the capital') → 1.0
    - If the answer contains any gazetteer entry → 0.8
    - If no entry found and the question is a wh-question that typically
      expects a named entity (who/where/what capital/which) → 0.2
    - Otherwise → 0.5 (neutral; factual check not applicable)
    """
    matches = find_matches(answer)
    if not matches:
        # Penalise only when the question clearly expects a named entity
        q_lower = question.lower()
        expects_entity = any(
            q_lower.startswith(w)
            for w in ("who ", "where ", "which country", "which city",
                      "what country", "what city", "what capital",
                      "what is the capital", "what language")
        )
        return 0.2 if expects_entity else 0.5

    matched_categories = {cat for _, cat in matches}

    # Broad heuristics: does the question type align with a matched category?
    q_lower = question.lower()
    if q_lower.startswith("who") and "PERSON" in matched_categories:
        return 1.0
    if q_lower.startswith("where") and matched_categories & {"CAPITAL", "CITY", "COUNTRY", "LANDMARK"}:
        return 1.0
    if ("capital" in q_lower or "city" in q_lower) and matched_categories & {"CAPITAL", "CITY"}:
        return 1.0
    if "language" in q_lower and "LANGUAGE" in matched_categories:
        return 1.0
    if "currency" in q_lower and "CURRENCY" in matched_categories:
        return 1.0
    if ("country" in q_lower or "nation" in q_lower) and "COUNTRY" in matched_categories:
        return 1.0
    if ("planet" in q_lower or "ocean" in q_lower or "mountain" in q_lower or "river" in q_lower) \
            and matched_categories & {"CELESTIAL", "LANDMARK"}:
        return 1.0

    # Any gazetteer hit is still a good signal
    return 0.8
