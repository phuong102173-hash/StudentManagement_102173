import re
import unicodedata


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize(text)))


def sentences(text: str) -> list[str]:
    # Preserve decimals; no language model needed. Not an atomic-fact extractor.
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]


def retrieve(claim: str, context: str, k: int = 4) -> list[str]:
    chunks = sentences(context)
    query = tokens(claim)
    return sorted(chunks, key=lambda c: len(query & tokens(c)) / max(1, len(query)), reverse=True)[:k]


def has_negation(text: str) -> bool:
    return bool(tokens(text) & {"không", "chưa", "not", "never", "no", "neither", "cannot"})
