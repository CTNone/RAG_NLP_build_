import argparse
import csv
import json
import time
from pathlib import Path

from src import load_chroma_db, load_config, retrieve_documents
from src.llm import call_llm_api, create_prompt
from src.eval.matching import match_answer


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_present(row, candidates):
    for candidate in candidates:
        value = row.get(candidate)
        if value:
            return value
    return ""


def _load_questions(path: Path) -> list[str]:
    # Prefer DictReader; fallback to first-column CSV if headers are missing.
    try:
        rows = load_rows(path)
        questions = []
        for row in rows:
            q = first_present(row, ["Question", "Questions", "question", "questions"])
            if q:
                questions.append(q)
        if questions:
            return questions
    except Exception:
        pass

    questions = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            questions.append(row[0])
    return questions


def _load_golden_answers(golden_path: Path) -> dict[str, dict]:
    """Load golden file with columns: Question, Expected (or ExpectedRegex)."""
    if not golden_path.exists():
        return {}
    rows = load_rows(golden_path)
    mapping: dict[str, dict] = {}
    for row in rows:
        question = first_present(row, ["Question", "Questions", "question", "questions"])
        expected = first_present(row, ["Expected", "expected"])
        expected_regex = first_present(row, ["ExpectedRegex", "expected_regex", "regex"])
        if question:
            mapping[question] = {"expected": expected, "expected_regex": expected_regex}
    return mapping


def evaluate_file(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    golden_path: Path | None = None,
):
    config = load_config()
    db = load_chroma_db(config)

    questions = _load_questions(input_path)
    if limit:
        questions = questions[:limit]

    golden = _load_golden_answers(golden_path) if golden_path else {}

    results = []
    for question in questions:
        golden_row = golden.get(question, {})
        expected = golden_row.get("expected", "")
        expected_regex = golden_row.get("expected_regex", "")
        started = time.perf_counter()
        contexts = retrieve_documents(db, question, config["retriever_k"], config=config)
        prompt = create_prompt(question, contexts, config["template"], config=config)
        answer = call_llm_api(prompt, config, None)
        latency = time.perf_counter() - started
        passed = match_answer(answer, expected, expected_regex)

        citations = []
        for c in contexts:
            if isinstance(c, dict):
                citations.append(
                    {
                        "citation_id": c.get("citation_id"),
                        "title": c.get("title"),
                        "url": c.get("url"),
                        "section": c.get("section"),
                        "chunk_id": c.get("chunk_id"),
                        "score": c.get("score"),
                    }
                )

        results.append(
            {
                "question": question,
                "expected": expected,
                "expected_regex": expected_regex,
                "answer": answer,
                "latency_seconds": round(latency, 3),
                "contexts": contexts,
                "citations": citations,
                "pass": passed,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation and generate reports.")
    parser.add_argument("--input", help="CSV file with questions (optional when --all).")
    parser.add_argument("--all", action="store_true", help="Evaluate all CSVs in data/test_questions.")
    parser.add_argument("--golden-dir", default="data/test_goldens", help="Directory containing golden CSVs.")
    parser.add_argument("--output-dir", default="data/test_answers/eval", help="Output directory for reports.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    questions_dir = Path(config.get("test_questions_dir", "data/test_questions"))
    output_dir = Path(args.output_dir)
    golden_dir = Path(args.golden_dir)

    if args.all:
        summary = []
        for input_path in sorted(questions_dir.glob("*.csv")):
            golden_path = golden_dir / input_path.name
            output_path = output_dir / f"eval_{input_path.stem}.json"
            evaluate_file(input_path, output_path, args.limit, golden_path=golden_path)
            summary.append({"input": str(input_path), "output": str(output_path), "golden": str(golden_path)})
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    if not args.input:
        raise SystemExit("Provide --input or use --all")

    input_path = Path(args.input)
    golden_path = (golden_dir / input_path.name) if golden_dir else None
    output_path = output_dir / f"eval_{input_path.stem}.json"
    evaluate_file(input_path, output_path, args.limit, golden_path=golden_path)


if __name__ == "__main__":
    main()
