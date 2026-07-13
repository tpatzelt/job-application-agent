from __future__ import annotations

import re

# Language names as users enter them (or as intake extraction returns them),
# mapped to the canonical lowercase English name used in preferences.
LANGUAGE_ALIASES: dict[str, str] = {
    "english": "english",
    "englisch": "english",
    "en": "english",
    "german": "german",
    "deutsch": "german",
    "de": "german",
    "french": "french",
    "französisch": "french",
    "franzoesisch": "french",
    "français": "french",
    "francais": "french",
    "fr": "french",
    "spanish": "spanish",
    "spanisch": "spanish",
    "español": "spanish",
    "espanol": "spanish",
    "es": "spanish",
    "dutch": "dutch",
    "nederlands": "dutch",
    "niederländisch": "dutch",
    "niederlaendisch": "dutch",
    "nl": "dutch",
    "italian": "italian",
    "italienisch": "italian",
    "italiano": "italian",
    "it": "italian",
    "portuguese": "portuguese",
    "portugiesisch": "portuguese",
    "português": "portuguese",
    "portugues": "portuguese",
    "pt": "portuguese",
    "polish": "polish",
    "polnisch": "polish",
    "polski": "polish",
    "pl": "polish",
}

# Answers that mean "no preference" — the caller falls back to the language
# the user's own input is written in.
NO_PREFERENCE_ANSWERS: frozenset[str] = frozenset(
    {"", "-", "any", "none", "no", "no preference", "doesn't matter",
     "does not matter", "egal", "beliebig", "skip", "/skip"}
)

# ISO 639-1 codes for the Brave Search `search_lang` parameter, keyed by
# canonical language name.
LANGUAGE_CODES: dict[str, str] = {
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "dutch": "nl",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
}

# Distinctive function words per language, for detecting which language a
# user's documents are written in. Deliberately small: a CV or a couple of
# chat messages contain plenty of these.
_STOPWORDS: dict[str, frozenset[str]] = {
    "english": frozenset(
        "the and of to for with on that this are is was have from your "
        "our will at as by".split()
    ),
    "german": frozenset(
        "und der die das nicht mit für ist ich eine ein auf als auch "
        "werden wir bei oder sind dem den einer über durch nach zur "
        "zum im am".split()
    ),
    "french": frozenset(
        "le la les et une du que pour dans est vous nous avec sur par "
        "pas au aux ce cette".split()
    ),
    "spanish": frozenset(
        "el los las y de que en para con una por del se su es al como "
        "más este".split()
    ),
    "dutch": frozenset(
        "de het een en van voor met dat niet zijn op je wij bij naar "
        "ook worden onze".split()
    ),
    "italian": frozenset(
        "il di che per con una del non sono alla nel gli più questo "
        "della delle".split()
    ),
}

# Below this many stopword hits the text is too short/ambiguous to call.
_MIN_HITS = 3


def normalize_language(value: str | None) -> str:
    """Canonical lowercase language name for a user-entered value.

    "No preference" style answers normalize to "" so callers fall back to
    the detected input language; unknown languages pass through lowercased
    (the LLM query prompt can still use them verbatim)."""
    cleaned = str(value or "").strip().lower().strip(".!")
    if cleaned in NO_PREFERENCE_ANSWERS:
        return ""
    return LANGUAGE_ALIASES.get(cleaned, cleaned)


def language_code_for(language: str | None) -> str | None:
    """ISO 639-1 code for the Brave Search `search_lang` parameter."""
    return LANGUAGE_CODES.get(normalize_language(language))


def detect_language(text: str) -> str:
    """Best-guess canonical language name for a piece of user-written text
    (stopword counting), or "" when there isn't enough signal."""
    words = re.findall(r"[a-zà-öø-ÿœß]+", text.lower())
    if not words:
        return ""
    best_language = ""
    best_hits = 0
    for language, stopwords in _STOPWORDS.items():
        hits = sum(1 for word in words if word in stopwords)
        if hits > best_hits:
            best_language, best_hits = language, hits
    return best_language if best_hits >= _MIN_HITS else ""
