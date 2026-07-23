import pytest
from app import resolve_intent

CONFUSABLE_CASES = [
    ("lone", "education_loan"),                 # should fix to 'loan' and match education_loan
    ("provide any lone", "education_loan"),     # should fix to 'loan' and match education_loan
    ("fee", "fees"),                            # near-duplicates that should map to the same tag
    ("libary timing", "library"),               # common typo
    ("very strange gibberish word xyz", None),  # should NOT confidently match any intent
]

def test_confusable_pairs():
    for msg, expected_tag in CONFUSABLE_CASES:
        tag, resp, clarify, is_clarify = resolve_intent(msg)
        if expected_tag is None:
            assert tag in ("clarify", "unknown", "document_retrieval"), f"{msg!r} should not confidently resolve, got {tag}"
        else:
            assert tag == expected_tag, f"{msg!r} expected {expected_tag}, got {tag}"
