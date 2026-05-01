import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
FRONT_MATTER_RE = re.compile(r"^---\s*$")

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "so",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "up",
    "us",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    return text


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def extract_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [piece.strip() for piece in pieces if piece.strip()]
