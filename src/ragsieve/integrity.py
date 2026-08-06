from __future__ import annotations

import unicodedata
from collections import Counter
from itertools import pairwise


def character_script(character: str) -> str | None:
    """Map an alphabetic Unicode character to a coarse writing-system label."""
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    if not name:
        return "OTHER"
    for script in (
        "LATIN",
        "CYRILLIC",
        "ARABIC",
        "HEBREW",
        "DEVANAGARI",
        "BENGALI",
        "THAI",
        "ARMENIAN",
        "GEORGIAN",
        "GREEK",
    ):
        if script in name:
            return script
    if any(token in name for token in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
        return "EAST_ASIAN"
    return "OTHER"


def script_anomaly(text: str) -> dict[str, float]:
    """Return mixed-script mass and script-transition rate for one document."""
    scripts = [script for character in text if (script := character_script(character)) is not None]
    if not scripts:
        return {"mixed_script_fraction": 0.0, "script_transition_rate": 0.0}
    counts = Counter(scripts)
    transitions = sum(left != right for left, right in pairwise(scripts))
    return {
        "mixed_script_fraction": 1.0 - max(counts.values()) / len(scripts),
        "script_transition_rate": transitions / max(1, len(scripts) - 1),
    }
