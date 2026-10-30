"""Print Markdown summary tables from the latest saved evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from helpdesk.evaluate import KS, QuestionResult, summarise

RESULTS = Path(__file__).resolve().parents[1] / "result"
PRED_REPO = "prediction-markets"


def latest_runs(results_dir: str | Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        name = config.get("name")
        if not isinstance(name, str):
            continue
        latest[name] = payload
    return latest


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _summary_rows(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for name, payload in sorted(runs.items()):
        overall = payload.get("summary", {}).get("overall", {})
        rows.append(
            [
                name,
                str(overall.get("n", 0)),
                *[_fmt(overall.get(f"recall@{k}")) for k in KS],
                _fmt(overall.get("mrr")),
            ]
        )
    return rows


def _results_of_run(
    payload: dict[str, Any], only_repo: str | None = None
) -> list[QuestionResult]:
    results = []
    for row in payload.get("questions", []):
        expected = [
            (item["repo"], item["path"])
            if isinstance(item, dict)
            else tuple(item)
            for item in row.get("expected", [])
        ]
        negative = row.get("is_negative", row.get("type") == "negative")
        repos = row.get("repos") or sorted({repo for repo, _ in expected})
        if only_repo and not negative and only_repo not in repos:
            continue
        results.append(
            QuestionResult(
                id=row["id"],
                type=row["type"],
                repos=repos,
                expected=expected,
                retrieved=row.get("retrieved", []),
                rank=row.get("rank"),
                all_found_rank=row.get("all_found_rank"),
                is_negative=negative,
            )
        )
    return results


def _detail(name: str, results: list[QuestionResult]) -> None:
    summary = summarise(results)
    print(f"## {name}: by type")
    type_rows = []
    for group, metrics in sorted(summary.get("by_type", {}).items()):
        type_rows.append(
            [
                group,
                str(metrics.get("n", 0)),
                *[_fmt(metrics.get(f"recall@{k}")) for k in KS],
                _fmt(metrics.get("mrr")),
            ]
        )
    _table(["type", "n", *[f"R@{k}" for k in KS], "MRR"], type_rows)

    print(f"\n## {name}: by repository")
    repo_rows = []
    for group, metrics in sorted(summary.get("by_repo", {}).items()):
        repo_rows.append(
            [
                group,
                str(metrics.get("n", 0)),
                *[_fmt(metrics.get(f"recall@{k}")) for k in KS],
                _fmt(metrics.get("mrr")),
            ]
        )
    _table(["repo", "n", *[f"R@{k}" for k in KS], "MRR"], repo_rows)

    print(f"\n## {name}: multi-file and negative scores")
    multi = summary.get("multi_all_found", {})
    negatives = summary.get("top1_score", {}).get("negative")
    _table(
        ["metric", *[f"@{k}" for k in KS]],
        [["multi all-files found", *[_fmt(multi.get(str(k))) for k in KS]]],
    )
    if negatives:
        print("\n### Negative top-1 score spread")
        _table(
            ["n", "min", "median", "max"],
            [[
                str(negatives["n"]),
                _fmt(negatives["min"]),
                _fmt(negatives["median"]),
                _fmt(negatives["max"]),
            ]],
        )

    misses = [
        result for result in results
        if not result.is_negative
        and (result.rank is None or result.rank > 1)
    ]
    if misses:
        print(f"\n### Misses / non-top-1 ({len(misses)})")
        for result in sorted(misses, key=lambda item: item.rank or 10**9)[:10]:
            top = result.retrieved[0] if result.retrieved else {}
            print(
                f"- {result.id} rank={result.rank} "
                f"expected={result.expected} top={top.get('path')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail",
        help="show per-type and per-repository detail (default: best full-set MRR)",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    runs = latest_runs(args.results_dir)
    if not runs:
        print(f"No saved evaluation runs found in {args.results_dir}")
        return

    print("# Retrieval evaluation")
    _table(
        ["config", "n", *[f"R@{k}" for k in KS], "MRR"],
        _summary_rows(runs),
    )

    full_runs = {
        name: payload
        for name, payload in runs.items()
        if not payload.get("config", {}).get("only_repo")
    }
    comparison_runs = {
        name: summarise(_results_of_run(payload, PRED_REPO))
        for name, payload in full_runs.items()
    }
    pred_run = runs.get("pred-v1")
    if pred_run is not None:
        comparison_runs["pred-v1"] = summarise(_results_of_run(pred_run))
    if comparison_runs:
        print(f"\n## {PRED_REPO} questions only (fair comparison with pred-v1)")
        rows = []
        for name, summary in sorted(comparison_runs.items()):
            overall = summary["overall"]
            rows.append(
                [
                    name,
                    str(overall["n"]),
                    *[_fmt(overall.get(f"recall@{k}")) for k in KS],
                    _fmt(overall.get("mrr")),
                ]
            )
        _table(["config", "n", *[f"R@{k}" for k in KS], "MRR"], rows)

    selected_name = args.detail
    if selected_name is None and full_runs:
        selected_name = max(
            full_runs,
            key=lambda name: (
                full_runs[name].get("summary", {})
                .get("overall", {})
                .get("mrr")
                if full_runs[name].get("summary", {})
                .get("overall", {})
                .get("mrr") is not None
                else float("-inf"),
                name,
            ),
        )
    if selected_name is not None:
        payload = runs.get(selected_name)
        if payload is None:
            parser.error(f"No saved run found for config {selected_name!r}")
        print()
        _detail(selected_name, _results_of_run(payload))


if __name__ == "__main__":
    main()
