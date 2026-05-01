import math
from collections import Counter
from pathlib import Path
from models import Article
from text_utils import normalize_text, tokenize, HEADING_RE, FRONT_MATTER_RE


def derive_product_area(path: Path, company: str) -> str:
    parts = [
        part for part in path.with_suffix("").parts if part not in {"data", company}
    ]
    if not parts:
        return company.lower()

    if company == "claude":
        area_map = {
            "account-management": "account_management",
            "conversation-management": "conversation_management",
            "features-and-capabilities": "features_and_capabilities",
            "get-started-with-claude": "get_started",
            "personalization-and-settings": "personalization_and_settings",
            "troubleshooting": "troubleshooting",
            "usage-and-limits": "usage_and_limits",
            "claude-code": "claude_code",
            "pro-and-max-plans": "plans",
            "team-and-enterprise-plans": "team_and_enterprise",
            "claude-api-and-console": "api_and_console",
            "safeguards": "safeguards",
            "amazon-bedrock": "amazon_bedrock",
            "claude-desktop": "claude_desktop",
        }
        for part in parts[1:3]:
            if part in area_map:
                return area_map[part]
        return (
            area_map.get(parts[1], parts[1].replace("-", "_"))
            if len(parts) > 1
            else "claude"
        )

    if company == "hackerrank":
        area_map = {
            "general-help": "general_help",
            "integrations": "integrations",
            "interviews": "interviews",
            "screen": "screen",
            "skillup": "skillup",
            "settings": "settings",
            "engage": "engage",
            "chakra": "chakra",
            "library": "library",
            "community": "community",
            "hackerrank_community": "community",
            "uncategorized": "uncategorized",
        }
        for part in parts[1:3]:
            if part in area_map:
                return area_map[part]
        return (
            area_map.get(parts[1], parts[1].replace("-", "_"))
            if len(parts) > 1
            else "hackerrank"
        )

    if company == "visa":
        area_map = {
            "support": "support",
            "consumer": "consumer",
            "merchant": "merchant",
            "small-business": "small_business",
            "travel-support": "travel_support",
            "visa-rules": "visa_rules",
            "fraud-protection": "fraud_protection",
            "dispute-resolution": "dispute_resolution",
            "travelers-cheques": "travelers_cheques",
            "checkout-fees-contact-form": "checkout_fees",
        }
        for part in parts[1:4]:
            if part in area_map:
                return area_map[part]
        return (
            area_map.get(parts[1], parts[1].replace("-", "_"))
            if len(parts) > 1
            else "visa"
        )

    return parts[1].replace("-", "_") if len(parts) > 1 else company.lower()


def load_markdown_article(path: Path, data_root: Path) -> Article:
    raw = normalize_text(path.read_text(encoding="utf-8"))
    lines = raw.splitlines()
    title = path.stem
    breadcrumbs: list[str] = []
    body_lines: list[str] = []
    in_front_matter = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and FRONT_MATTER_RE.match(stripped):
            in_front_matter = True
            continue
        if in_front_matter:
            if FRONT_MATTER_RE.match(stripped):
                in_front_matter = False
                continue
            if stripped.startswith("title:") and title == path.stem:
                title = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            elif stripped.startswith("- ") and '"' in stripped:
                breadcrumbs.append(stripped.strip("- ").strip('"'))
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match and not body_lines:
            title = heading_match.group(1).strip()
            continue
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    company_index = len(data_root.parts)
    company = (
        path.parts[company_index] if len(path.parts) > company_index else "unknown"
    )
    product_area = derive_product_area(path, company)
    title_tokens = frozenset(tokenize(title))
    heading_tokens = frozenset(tokenize(" ".join(breadcrumbs) + " " + title))
    tokens = frozenset(tokenize(f"{title}\n{body}\n{' '.join(breadcrumbs)}"))
    return Article(
        path=path,
        company=company,
        product_area=product_area,
        title=title,
        breadcrumbs=tuple(breadcrumbs),
        body=body,
        tokens=tokens,
        heading_tokens=heading_tokens | title_tokens,
    )


def load_corpus(data_root: Path) -> list[Article]:
    articles: list[Article] = []
    for path in sorted(data_root.rglob("*.md")):
        if path.name.lower() == "index.md":
            continue
        articles.append(load_markdown_article(path, data_root))
    return articles


class BM25Retriever:
    def __init__(self, corpus: list[Article]):
        self.corpus = corpus
        self.doc_len: list[int] = []
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75
        
        self._initialize()

    def _initialize(self):
        nd = len(self.corpus)
        num_doc = 0

        frequencies = {}

        for article in self.corpus:
            doc_tokens = (
                list(article.tokens)
                + list(article.heading_tokens) * 2
                + list(tokenize(article.title)) * 3
            )
            self.doc_len.append(len(doc_tokens))
            num_doc += len(doc_tokens)

            freq = Counter(doc_tokens)
            self.doc_freqs.append(freq)

            for word in freq.keys():
                frequencies[word] = frequencies.get(word, 0) + 1

        self.avgdl = num_doc / nd if nd > 0 else 0

        for word, freq in frequencies.items():
            idf = math.log(1 + (nd - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for index in range(len(self.corpus)):
            score = 0.0
            doc_len = self.doc_len[index]
            freqs = self.doc_freqs[index]

            for token in query_tokens:
                if token not in freqs:
                    continue
                freq = freqs[token]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += (self.idf.get(token, 0) * num) / den
            scores.append(score)
        return scores

    def retrieve(
        self, query: str, top_k: int = 5, company: str | None = None
    ) -> list[Article]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.get_scores(query_tokens)

        ranked = []
        for index, score in enumerate(scores):
            article = self.corpus[index]

            # Hard filter based on company
            if company and company.lower() != "none" and article.company != company:
                continue

            # Penalize index files
            if article.path.name.lower().startswith("index"):
                score *= 0.6

            if score > 0:
                ranked.append((score, article))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in ranked[:top_k]]
