"""Safety unit tests — no DB, no LLM."""

from src.safety import redact_pii_text


def test_redacts_openrouter_key():
    text = "my key is sk-or-v1-1234567890abcdefghijabcdefghij please keep it secret"
    out = redact_pii_text(text)
    assert "sk-or" not in out
    assert "[REDACTED]" in out


def test_redacts_ssn():
    out = redact_pii_text("ssn 123-45-6789 is mine")
    assert "123-45-6789" not in out
    assert "[REDACTED]" in out


def test_passthrough():
    text = "what's my net worth right now?"
    assert redact_pii_text(text) == text


def test_empty():
    assert redact_pii_text("") == ""
