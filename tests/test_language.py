from __future__ import annotations

from src.language import detect_language, language_code_for, normalize_language

GERMAN_TEXT = (
    "Ich bin Softwareentwickler mit mehrjähriger Erfahrung und suche eine "
    "neue Stelle in München. Die Arbeit mit modernen Technologien ist mir "
    "wichtig, und ich möchte bei einem Unternehmen arbeiten, das auf "
    "Qualität setzt."
)

ENGLISH_TEXT = (
    "I am a software engineer with several years of experience and I am "
    "looking for a new role in Berlin. Working with modern technologies is "
    "important to me, and I want to join a company that values quality."
)


def test_detect_language_german() -> None:
    assert detect_language(GERMAN_TEXT) == "german"


def test_detect_language_english() -> None:
    assert detect_language(ENGLISH_TEXT) == "english"


def test_detect_language_insufficient_signal() -> None:
    assert detect_language("") == ""
    assert detect_language("Python developer CV") == ""


def test_normalize_language_aliases() -> None:
    assert normalize_language("Deutsch") == "german"
    assert normalize_language(" GERMAN. ") == "german"
    assert normalize_language("EN") == "english"
    assert normalize_language("nederlands") == "dutch"
    # Unknown languages pass through lowercased for the LLM prompt.
    assert normalize_language("Klingon") == "klingon"


def test_normalize_language_no_preference() -> None:
    for answer in ("", "any", "no preference", "/skip", "egal", None):
        assert normalize_language(answer) == ""


def test_language_code_for() -> None:
    assert language_code_for("german") == "de"
    assert language_code_for("Deutsch") == "de"
    assert language_code_for("english") == "en"
    assert language_code_for("klingon") is None
    assert language_code_for("") is None
