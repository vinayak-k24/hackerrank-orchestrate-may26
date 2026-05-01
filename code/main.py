import csv
import json
import os
from pathlib import Path

from config import bootstrap_environment, get_config
from models import Ticket, TriageResult
from retriever import load_corpus, BM25Retriever
from agent import SupportAgent

def read_input_rows(csv_path: Path) -> list[Ticket]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[Ticket] = []
        for row in reader:
            rows.append(Ticket(
                issue=(row.get("Issue", row.get("issue", "")) or "").strip(),
                subject=(row.get("Subject", row.get("subject", "")) or "").strip(),
                company=(row.get("Company", row.get("company", "")) or "").strip()
            ))
        return rows

def write_output_rows(csv_path: Path, results: list[TriageResult], inputs: list[Ticket]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ticket, result in zip(inputs, results):
            writer.writerow({
                "issue": ticket.issue,
                "subject": ticket.subject,
                "company": ticket.company or "None",
                "response": result.response,
                "product_area": result.product_area,
                "status": result.status,
                "request_type": result.request_type,
                "justification": result.justification,
            })

def find_input_and_output_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    support_dir = repo_root / "support_tickets"
    if not support_dir.exists():
        support_dir = repo_root / "support_issues"

    input_name = os.getenv("SUPPORT_INPUT_CSV", "support_tickets.csv")
    sample_name = os.getenv("SUPPORT_SAMPLE_CSV", "sample_support_tickets.csv")
    output_name = os.getenv("SUPPORT_OUTPUT_CSV", "output.csv")

    input_csv = support_dir / input_name
    if not input_csv.exists() and input_name == "support_tickets.csv":
        input_csv = support_dir / "support_issues.csv"

    sample_csv = support_dir / sample_name
    if not sample_csv.exists() and sample_name == "sample_support_tickets.csv":
        sample_csv = support_dir / "sample_support_issues.csv"

    return input_csv, sample_csv, support_dir / output_name

def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap_environment(repo_root)
    config = get_config()

    data_root = Path(os.getenv("SUPPORT_DATA_ROOT", str(repo_root / "data")))
    input_csv, sample_csv, output_csv = find_input_and_output_paths(repo_root)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    articles = load_corpus(data_root)
    retriever = BM25Retriever(articles)
    agent = SupportAgent(retriever, config)

    rows = read_input_rows(input_csv)
    results: list[TriageResult] = []
    
    for row in rows:
        results.append(agent.triage(row))

    write_output_rows(output_csv, results, rows)

    metadata = {
        "input_csv": str(input_csv),
        "sample_csv": str(sample_csv),
        "output_csv": str(output_csv),
        "articles_indexed": len(articles),
        "rows_processed": len(results),
    }
    print(json.dumps(metadata, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())