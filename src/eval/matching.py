import re
from typing import Optional


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def match_answer(
    answer: str,
    expected: str,
    expected_regex: str,
) -> Optional[bool]:
    """Return True/False if we can judge, else None when no golden provided."""
    if expected_regex:
        try:
            return re.search(expected_regex, answer or "", flags=re.IGNORECASE) is not None
        except re.error:
            return False
    if expected:
        return normalize_text(expected) in normalize_text(answer)
    return None

