"""Grade a model response against a ground-truth answer.

Integers, player names and team names must match exactly after normalization
(names allow small typos and partial names); percentages may be off by 0.5.
Scores ("<home team> <points>-<points> <away team>") need both teams with the right points.
"""

import re
import unicodedata

from rapidfuzz import fuzz

PCT_TOLERANCE = 0.5
NAME_MIN_RATIO = 90  # rapidfuzz ratio (0-100) above which two names are the same

FINAL_RE = re.compile(r"FINAL ANSWER\s*:\s*(.*)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d.,]*")
SCORE_RE = re.compile(r"^(.*?)(\d+)\s*(?:-|–|—|:|\bto\b)\s*(\d+)(.*)$", re.DOTALL)
NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def parse_final_answer(text: str) -> str | None:
    """Text after the last 'FINAL ANSWER:' in the response, or None if there is none."""
    matches = FINAL_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip().strip("*_`").strip()


def normalize_text(s: str) -> str:
    """Lowercase, strip accents, drop parentheticals and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    return " ".join(s.replace("-", " ").split())


def parse_number(s: str) -> float | None:
    """The single number in a string ('1,234', '51,6%', '20 points', 'twelve'); None if zero or several."""
    s = str(s).strip().lower()
    found = NUMBER_RE.findall(s)
    if len(found) != 1:
        words = [NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", s) if w in NUMBER_WORDS]
        return float(words[0]) if not found and len(words) == 1 else None
    num = found[0].rstrip(".,")
    if "," in num and "." in num:
        num = num.replace(",", "")  # 1,234.5
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})+", num):
        num = num.replace(",", "")  # 1,234
    else:
        num = num.replace(",", ".")  # 51,6
    try:
        return float(num)
    except ValueError:
        return None


def names_match(answer: str, expected: str) -> bool:
    a, e = normalize_text(answer), normalize_text(expected)
    if not a:
        return False
    if a == e or fuzz.ratio(a, e) >= NAME_MIN_RATIO:
        return True
    # Partial name ("Zaragoza" for "Casademont Zaragoza"), but not a longer string that merely contains it.
    a_tokens, e_tokens = a.split(), e.split()
    return len(a_tokens) <= len(e_tokens) and all(
        any(fuzz.ratio(t, et) >= NAME_MIN_RATIO for et in e_tokens) for t in a_tokens
    )


def parse_score(s: str) -> tuple[str, int, int, str] | None:
    """('BAXI Manresa', 41, 24, 'Coviran Granada') from 'BAXI Manresa 41-24 Coviran Granada'; None if not a score."""
    match = SCORE_RE.match(str(s).strip())
    if not match:
        return None
    left, a, b, right = match.groups()
    return left.strip(" ,:"), int(a), int(b), right.strip(" ,:.")


def scores_match(answer: str, expected: str) -> bool:
    """Both teams named with the right points. The teams may be given in either order."""
    got, want = parse_score(answer), parse_score(expected)
    if got is None or want is None:
        return False
    flipped = (got[3], got[2], got[1], got[0])
    return any(
        (g[1], g[2]) == (want[1], want[2]) and names_match(g[0], want[0]) and names_match(g[3], want[3])
        for g in (got, flipped)
    )


def grade(response: str, expected, answer_type: str) -> dict:
    """{'correct': bool, 'model_answer': parsed final answer or None}."""
    answer = parse_final_answer(response)
    if answer is None:
        return {"correct": False, "model_answer": None}
    if answer_type == "integer":
        value = parse_number(answer)
        correct = value is not None and value == int(expected)
    elif answer_type == "percentage":
        value = parse_number(answer)
        # A bare fraction like 0.516 is read as 51.6%.
        if value is not None and "%" not in answer and "percent" not in answer.lower() and value <= 1 < expected:
            value *= 100
        correct = value is not None and abs(value - float(expected)) <= PCT_TOLERANCE + 1e-9
    elif answer_type in ("player", "team"):
        correct = names_match(answer, expected)
    elif answer_type == "score":
        correct = scores_match(answer, expected)
    else:
        raise ValueError(f"Unknown answer_type {answer_type!r}")
    return {"correct": bool(correct), "model_answer": answer}
