try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None  # type: ignore
    APIError = Exception  # type: ignore

from models import Article, TriageResult, Ticket
from retriever import BM25Retriever
from text_utils import tokenize, extract_sentences

HIGH_RISK_KEYWORDS = {
    "fraud",
    "stolen",
    "steal",
    "identity theft",
    "compromised",
    "phishing",
    "chargeback",
    "dispute",
    "urgent",
    "immediately",
    "security breach",
    "private info",
    "personal data",
    "expiration",
    "start time",
    "end time",
    "password",
    "access lost",
    "rejected me",
    "review my answers",
    "increase my score",
    "move me to the next round",
    "site is down",
    "pages are inaccessible",
    "none of the pages",
}

UNSUPPORTED_KEYWORDS = {
    "actor",
    "movie",
    "sports",
    "weather",
    "music",
    "news",
    "politics",
    "recipe",
    "joke",
    "quote",
    "random",
}


class SupportAgent:
    def __init__(self, retriever: BM25Retriever, config: dict[str, str]):
        self.retriever = retriever
        self.config = config
        self.openai_client = self._init_openai_client()

    def _init_openai_client(self) -> OpenAI | None:
        if OpenAI is None:
            return None
        use_openai = self.config.get("USE_OPENAI", "false").lower() == "true"
        api_key = self.config.get("OPENAI_API_KEY", "").strip()
        if not use_openai or not api_key:
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def triage(self, ticket: Ticket) -> TriageResult:
        company = self._resolve_company(ticket)
        request_type = self._classify_request_type(ticket)

        # Determine topic manually for some specific keywords
        manual_area, manual_keywords = self._manual_topic_route(company, ticket)

        # Retrieve articles
        query = ticket.full_text + " " + " ".join(manual_keywords)
        articles = self.retriever.retrieve(query, top_k=5, company=company)
        best_article = articles[0] if articles else None

        if request_type == "invalid":
            return TriageResult(
                status="replied",
                product_area=manual_area or self._default_area(company),
                response=self._make_out_of_scope_response(),
                justification="The ticket is unrelated to the supported products, so a polite out-of-scope reply is safest.",
                request_type="invalid",
            )

        escalated = self._should_escalate(ticket, request_type, company, best_article)

        if best_article is None:
            if escalated:
                return TriageResult(
                    status="escalated",
                    product_area=manual_area or "support",
                    response="",
                    justification=self._build_justification(
                        company, request_type, best_article, True
                    ),
                    request_type=request_type,
                )
            return TriageResult(
                status="replied",
                product_area=manual_area or "support",
                response="I couldn’t find a strong corpus match, so please share more product details.",
                justification=self._build_justification(
                    company, request_type, best_article, False
                ),
                request_type=request_type,
            )

        if escalated:
            return TriageResult(
                status="escalated",
                product_area=best_article.product_area,
                response="",
                justification=self._build_justification(
                    company, request_type, best_article, True
                ),
                request_type=request_type,
            )

        response = self._generate_response(best_article, ticket)
        return TriageResult(
            status="replied",
            product_area=manual_area or best_article.product_area,
            response=response,
            justification=self._build_justification(
                company, request_type, best_article, False
            ),
            request_type=request_type,
        )

    def _resolve_company(self, ticket: Ticket) -> str | None:
        company = (
            ticket.company.lower()
            if ticket.company
            and ticket.company.strip()
            and ticket.company.strip().lower() != "none"
            else None
        )
        if company not in {"hackerrank", "claude", "visa"}:
            lowered = ticket.full_text.lower()
            if "hackerrank" in lowered:
                return "hackerrank"
            if "claude" in lowered:
                return "claude"
            if "visa" in lowered:
                return "visa"
            return None
        return company

    def _default_area(self, company: str | None) -> str:
        if company == "claude":
            return "conversation_management"
        if company == "hackerrank":
            return "general_help"
        if company == "visa":
            return "support"
        return "out_of_scope"

    def _classify_request_type(self, ticket: Ticket) -> str:
        text = ticket.full_text.lower()
        feature_markers = {
            "feature request",
            "can you add",
            "could you add",
            "would love",
            "i wish",
            "please add",
            "new feature",
            "enhancement",
            "improve",
        }
        if any(marker in text for marker in feature_markers):
            return "feature_request"
        if any(
            marker in text
            for marker in [
                "not working",
                "stopped",
                "broken",
                "error",
                "down",
                "failed",
                "bug",
                "glitch",
                "cannot",
                "can't",
                "won't",
            ]
        ):
            return "bug"
        if self._looks_like_invalid(text):
            return "invalid"
        return "product_issue"

    def _looks_like_invalid(self, text: str) -> bool:
        if not text.strip():
            return True
        if any(term in text for term in UNSUPPORTED_KEYWORDS):
            return True
        if (
            len(tokenize(text)) < 3
            and "hackerrank" not in text
            and "claude" not in text
            and "visa" not in text
        ):
            return True
        return False

    def _should_escalate(
        self,
        ticket: Ticket,
        request_type: str,
        company: str | None,
        article: Article | None,
    ) -> bool:
        text = ticket.full_text.lower()
        if (
            "site is down" in text
            or "pages are inaccessible" in text
            or "none of the pages" in text
        ):
            return True

        if any(keyword in text for keyword in HIGH_RISK_KEYWORDS):
            article_title = article.title.lower() if article else ""
            if company == "visa" and any(
                term in text for term in ["stolen", "lost", "fraud", "dispute"]
            ):
                return False
            if company == "claude" and (
                any(
                    term in text
                    for term in ["removed my seat", "workspace owner", "admin"]
                )
                or any(
                    term in article_title
                    for term in ["incognito", "delete or rename", "conversation"]
                )
            ):
                return False
            if company == "hackerrank" and "refund" in text:
                return False
            if (
                company == "hackerrank"
                and article_title
                and any(
                    term in article_title
                    for term in ["modify test expiration time", "extend test duration"]
                )
            ):
                return False
            return True

        if request_type == "bug" and article is None:
            return True
        return False

    def _manual_topic_route(
        self, company: str | None, ticket: Ticket
    ) -> tuple[str | None, list[str]]:
        text = ticket.full_text.lower()
        if company == "hackerrank":
            if "mock interview" in text:
                return "subscriptions_payments_and_billing", [
                    "purchase mock interview credits",
                    "subscriptions payments and billing faqs",
                ]
            if any(
                term in text
                for term in ["extra time", "reinvite", "re-invite", "accommodation"]
            ):
                return "screen", ["extend test duration for candidates"]
            if "variant" in text:
                return "screen", ["create test variants"]
            if any(
                term in text
                for term in [
                    "test active",
                    "stay active",
                    "remain active",
                    "how long",
                    "expire",
                    "expired",
                    "invitation",
                    "start time",
                    "end time",
                ]
            ):
                return "screen", ["modify test expiration time"]
            if any(term in text for term in ["delete account", "account delete"]):
                return "general_help", ["contact hackerrank support"]

        if company == "claude":
            if any(
                term in text
                for term in ["private info", "delete", "conversation", "incognito"]
            ):
                return "conversation_management", [
                    "delete or rename a conversation",
                    "using incognito chats",
                ]
            if any(
                term in text
                for term in [
                    "seat removed",
                    "workspace owner",
                    "admin",
                    "access lost",
                    "log in",
                ]
            ):
                return "account_management", [
                    "how to get support",
                    "logging in to your claude account",
                ]
            if any(
                term in text for term in ["refund", "cancel subscription", "billing"]
            ):
                return "plans", ["requesting a refund for a paid claude plan"]

        if company == "visa":
            if any(
                term in text for term in ["stolen", "lost card", "compromised", "fraud"]
            ):
                return "consumer", ["visa travel services", "fraud prevention"]
            if any(
                term in text for term in ["traveller", "traveler", "cheque", "cheques"]
            ):
                return "travelers_cheques", ["visa traveller's cheques"]
            if any(
                term in text
                for term in [
                    "dispute",
                    "refund me",
                    "wrong product",
                    "merchant",
                    "charge",
                ]
            ):
                return "dispute_resolution", [
                    "dispute resolution",
                    "visa credit card rules",
                ]
            if any(term in text for term in ["rules", "regulations", "policy"]):
                return "visa_rules", ["visa credit card rules"]

        return None, []

    def _generate_response(self, article: Article, ticket: Ticket) -> str:
        if self.openai_client is not None:
            prompt = (
                f"You are a helpful support agent. Based on the support article below, "
                f"provide a clear, concise answer to the customer's issue. "
                f"Only use information from the article; do not make up details.\n\n"
                f"Customer Issue: {ticket.issue}\n"
                f"Subject: {ticket.subject}\n\n"
                f"Support Article:\n{article.body[:2000]}\n\n"
                f"Answer:"
            )
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=float(self.config.get("OPENAI_TEMPERATURE", "0.3")),
                    max_tokens=int(self.config.get("OPENAI_MAX_TOKENS", "250")),
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception:
                pass

        # Dynamic Extractive Summarization Fallback
        # Instead of hardcoded strings, we dynamically score and extract the most relevant
        # sentences from the article body based on the issue query.
        sentences = extract_sentences(article.body)
        if not sentences:
            return f"According to our documentation in '{article.title}', you can find steps to resolve this. Please follow the standard procedure described there."

        issue_tokens = set(tokenize(ticket.full_text))

        # Score each sentence by its relevance to the query and its length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_tokens = set(tokenize(sentence))
            # Calculate Jaccard similarity for relevance
            intersection = len(sentence_tokens & issue_tokens)
            union = len(sentence_tokens | issue_tokens)
            relevance = (intersection / union) if union > 0 else 0

            # Boost score if sentence is near the beginning (often contains main points)
            position_boost = 1.0 / (i + 1.0)

            final_score = (relevance * 5.0) + position_boost
            scored_sentences.append((final_score, sentence))

        # Sort by score and take top 2 sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        best_sentences = [s for score, s in scored_sentences[:2] if score > 0.0]

        # If we couldn't find a strong match, just return the first few lines of the article
        if not best_sentences:
            best_sentences = sentences[:2]

        # Format cleanly
        cleaned = []
        for sentence in best_sentences:
            sentence = sentence.strip()
            if len(sentence) > 200:
                sentence = sentence[:197].rsplit(" ", 1)[0] + "..."
            cleaned.append(sentence)

        return " ".join(cleaned)

    def _build_justification(
        self,
        company: str | None,
        request_type: str,
        article: Article | None,
        escalated: bool,
    ) -> str:
        if escalated:
            if request_type == "invalid":
                return "The request is out of scope or unsupported, so it should not be answered as a product support case."
            return "The issue is high-risk, unsupported, or needs human review, so it should be escalated instead of guessed."

        if article is None:
            return "No strong corpus match was found, so the safest response is a brief out-of-scope reply."

        return f"Matched the most relevant {article.company} support article in the {article.product_area} area and answered only with corpus-grounded guidance."

    def _make_out_of_scope_response(self) -> str:
        return "I’m sorry, but this request is outside the support corpus I can safely answer. Please contact the relevant support team or provide a product-related question."
