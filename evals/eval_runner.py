import json
from pathlib import Path

from agents.career_agent import match_resume_to_jd
from rag.rag_chain import retrieve_sources


BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "data" / "eval"


def load_json(filename: str) -> list[dict]:
    path = EVAL_DIR / filename
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_rag_eval(top_k: int = 3) -> dict:
    cases = load_json("rag_cases.json")
    results = []

    for case in cases:
        sources, retriever = retrieve_sources(case["query"], top_k=top_k)
        returned_titles = [source["title"] for source in sources]
        passed = case["expected_title"] in returned_titles

        results.append({
            "query": case["query"],
            "expected_title": case["expected_title"],
            "returned_titles": returned_titles,
            "passed": passed,
            "retriever": retriever,
        })

    passed_count = sum(1 for result in results if result["passed"])

    return {
        "name": "rag_retrieval_eval",
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "accuracy": round(passed_count / len(results), 4) if results else 0,
        "results": results,
    }


def run_career_eval() -> dict:
    cases = load_json("career_cases.json")
    results = []

    for case in cases:
        matched = match_resume_to_jd(
            resume_text=case["resume_text"],
            jd_text=case["jd_text"],
        )
        missing_keywords = matched["missing_keywords"]
        passed = sorted(missing_keywords) == sorted(case["expected_missing"])

        results.append({
            "expected_missing": case["expected_missing"],
            "actual_missing": missing_keywords,
            "match_score": matched["match_score"],
            "passed": passed,
        })

    passed_count = sum(1 for result in results if result["passed"])

    return {
        "name": "career_match_eval",
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "accuracy": round(passed_count / len(results), 4) if results else 0,
        "results": results,
    }


def run_all_evals() -> dict:
    rag_eval = run_rag_eval()
    career_eval = run_career_eval()

    total = rag_eval["total"] + career_eval["total"]
    passed = rag_eval["passed"] + career_eval["passed"]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(passed / total, 4) if total else 0,
        "suites": [
            rag_eval,
            career_eval,
        ],
    }
