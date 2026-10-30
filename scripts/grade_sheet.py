"""Write a Markdown faithfulness grading sheet from a saved answer run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "results" / "answers"


def latest_run(directory: Path, generator: str) -> Path:
    paths = sorted(directory.glob(f"*_{generator}.json"))
    if not paths:
        raise FileNotFoundError(
            f"No saved {generator} answer run found in {directory}"
        )
    return paths[-1]


def _format_expected(expected: list[dict[str, str]]) -> str:
    if not expected:
        return "No expected source (negative question)"
    return ", ".join(f"{item['repo']}/{item['path']}" for item in expected)


def render_sheet(payload: dict[str, Any], source_path: Path) -> str:
    config = payload.get("config", {})
    lines = [
        f"# Answer grading sheet: {config.get('generator', 'unknown')}",
        "",
        f"- Run: `{source_path.name}`",
        f"- Generator model: `{config.get('generator_model') or 'not recorded'}`",
        f"- Embedding model: `{config.get('embedding_model') or 'not recorded'}`",
        "",
        "For each answer, mark whether the answer is supported by the cited "
        "retrieved text. Also check whether the answer abstains correctly for "
        "negative questions.",
        "",
    ]
    for row in payload.get("questions", []):
        answer = row.get("answer", {})
        lines.extend(
            [
                f"## {row.get('id', 'unknown')} ({row.get('type', 'unknown')})",
                "",
                f"**Question:** {row.get('question', '')}",
                "",
                f"**Expected:** {_format_expected(row.get('expected', []))}",
                "",
                f"**Answered:** {'yes' if answer.get('answered') else 'no'}",
                "",
                f"**Answer:** {answer.get('answer', '') or '(empty)'}",
                "",
                "**Citations:**",
                "",
            ]
        )
        citations = answer.get("citations", [])
        if not citations:
            lines.extend(["- None", ""])
        for citation in citations:
            index = citation.get("index")
            retrieved = row.get("retrieved", [])
            hit = (
                retrieved[index - 1]
                if isinstance(index, int) and 1 <= index <= len(retrieved)
                else None
            )
            lines.extend(
                [
                    f"- [{index}] {citation.get('repo')}/{citation.get('path')}:"
                    f"{citation.get('line_start')}-{citation.get('line_end')}",
                ]
            )
            if hit:
                lines.extend(
                    [
                        "",
                        "```text",
                        str(hit.get("text", "")).strip(),
                        "```",
                    ]
                )
            lines.append("")
        lines.extend(
            [
                "**Manual grade:** [ ] Supported  [ ] Partially supported  "
                "[ ] Unsupported  [ ] Correct refusal",
                "",
                "**Reviewer notes:**",
                "",
                "________________________________________________________________",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=("haiku", "sonnet", "extractive"), default="haiku")
    parser.add_argument("--run", type=Path, help="saved answer JSON; defaults to latest generator run")
    parser.add_argument("--answers-dir", type=Path, default=ANSWERS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_path = args.run or latest_run(args.answers_dir, args.generator)
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    output_path = args.output or args.answers_dir / f"grading_{run_path.stem}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_sheet(payload, run_path), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
