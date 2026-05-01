from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Article:
    path: Path
    company: str
    product_area: str
    title: str
    breadcrumbs: tuple[str, ...]
    body: str
    tokens: frozenset[str]
    heading_tokens: frozenset[str]

@dataclass(frozen=True)
class TriageResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

@dataclass(frozen=True)
class Ticket:
    issue: str
    subject: str
    company: str | None

    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.issue}".strip()
