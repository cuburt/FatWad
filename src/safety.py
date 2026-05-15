"""Tiny PII redactor used in two places: the safety node (sanitises the inbound
user message) and the structlog pipeline (sanitises log records).

Kept dependency-free on purpose so tests can exercise it without standing up
FastAPI or the database. Patterns target the secrets a wealth app is likely to
see in chat input — API keys and credit-card-shaped digit runs.
"""

import re
from typing import Pattern

API_KEY_PATTERN: Pattern[str] = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|or-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,})\b"
)
CARD_PATTERN: Pattern[str] = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SSN_PATTERN: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

REDACTION = "[REDACTED]"


def redact_pii_text(text: str) -> str:
    if not text:
        return text
    text = API_KEY_PATTERN.sub(REDACTION, text)
    text = SSN_PATTERN.sub(REDACTION, text)
    text = CARD_PATTERN.sub(REDACTION, text)
    return text


def redact_pii_processor(logger, log_method, event_dict: dict) -> dict:
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = redact_pii_text(value)
    return event_dict
