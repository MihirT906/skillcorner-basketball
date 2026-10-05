import pytest

from grade import grade, parse_final_answer


def test_parse_final_answer():
    assert parse_final_answer("Alonso had 33.\nFINAL ANSWER: 33") == "33"
    assert parse_final_answer("blah\n**Final answer:**  Marco Spissu ") == "Marco Spissu"
    assert parse_final_answer("FINAL ANSWER: 1\nwait\nFINAL ANSWER: 2") == "2"
    assert parse_final_answer("The answer is 33.") is None


@pytest.mark.parametrize("answer, expected, answer_type, correct", [
    ("33", 33, "integer", True),
    ("33 points", 33, "integer", True),
    ("33.0", 33, "integer", True),
    ("1,234", 1234, "integer", True),
    ("twelve", 12, "integer", True),
    ("0", 0, "integer", True),
    ("34", 33, "integer", False),
    ("9/15", 9, "integer", False),  # two numbers: ambiguous
    ("unknown", 33, "integer", False),
    ("51.6%", 51.6, "percentage", True),
    ("51.6", 51.6, "percentage", True),
    ("51,6 %", 51.6, "percentage", True),
    ("52", 51.6, "percentage", True),  # within 0.5
    ("51.1 percent", 51.6, "percentage", True),
    ("0.516", 51.6, "percentage", True),
    ("52.2%", 51.6, "percentage", False),
    ("50%", 51.6, "percentage", False),
    ("Francisco Alonso", "Francisco Alonso", "player", True),
    ("francisco  alonso.", "Francisco Alonso", "player", True),
    ("Francísco Alonso", "Francisco Alonso", "player", True),
    ("Fransisco Alonso", "Francisco Alonso", "player", True),  # typo
    ("Alonso", "Francisco Alonso", "player", True),
    ("Francisco Alonso (33 points)", "Francisco Alonso", "player", True),
    ("Justin Wright Foreman", "Justin Wright-Foreman", "player", True),
    ("DeWayne Russell", "Francisco Alonso", "player", False),
    ("Zaragoza", "Casademont Zaragoza", "team", True),
    ("CASADEMONT ZARAGOZA", "Casademont Zaragoza", "team", True),
    ("Leche Rio Breogan", "Casademont Zaragoza", "team", False),
    ("Leche Rio Breogan or Casademont Zaragoza", "Casademont Zaragoza", "team", False),
    ("BAXI Manresa 41-24 Coviran Granada", "BAXI Manresa 41-24 Coviran Granada", "score", True),
    ("baxi manresa 41 - 24 coviran granada.", "BAXI Manresa 41-24 Coviran Granada", "score", True),
    ("BAXI Manresa 41–24 Coviran Granada", "BAXI Manresa 41-24 Coviran Granada", "score", True),  # en dash
    ("Manresa 41-24 Granada", "BAXI Manresa 41-24 Coviran Granada", "score", True),
    ("BAXI Manresa 41 to 24 Coviran Granada", "BAXI Manresa 41-24 Coviran Granada", "score", True),
    ("Coviran Granada 24-41 BAXI Manresa", "BAXI Manresa 41-24 Coviran Granada", "score", True),  # away first, same score
    ("BAXI Manresa 24-41 Coviran Granada", "BAXI Manresa 41-24 Coviran Granada", "score", False),  # points swapped
    ("Coviran Granada 41-24 BAXI Manresa", "BAXI Manresa 41-24 Coviran Granada", "score", False),
    ("BAXI Manresa 41-25 Coviran Granada", "BAXI Manresa 41-24 Coviran Granada", "score", False),
    ("41-24", "BAXI Manresa 41-24 Coviran Granada", "score", False),  # no teams
    ("BAXI Manresa", "BAXI Manresa 41-24 Coviran Granada", "score", False),
    ("Malaga 48-48 Valencia BC", "Malaga 48-48 Valencia BC", "score", True),
])
def test_grade(answer, expected, answer_type, correct):
    result = grade(f"Some reasoning.\nFINAL ANSWER: {answer}", expected, answer_type)
    assert result["correct"] is correct
    assert result["model_answer"] == answer


def test_missing_final_answer_is_wrong():
    assert grade("It was 33.", 33, "integer") == {"correct": False, "model_answer": None}
