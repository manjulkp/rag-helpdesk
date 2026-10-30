"""Do answers flag conflicting sources? (data/golden_conflicts.json)

    python -m scripts.eval_conflicts

The corpus has no genuine conflicting documents, so the conflict cases are
synthetic: a real chunk, plus a copy with one number changed under another
path and date (older, newer, or both unknown). Controls have no conflict and
must not get one invented. Haiku answers each; the script checks whether the
answer flags the conflict and contains the current value, and writes a sheet
in results/conflicts/ for grading by hand.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from helpdesk.embedder import DEFAULT_MODEL
from helpdesk.generate import ClaudeGenerator
from helpdesk.index import DEFAULT_PERSIST_DIR, SearchHit, get_collection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "conflicts"
FLAG = re.compile(r"conflict|disagree|differ|inconsisten|contradict|two sources|older|outdated|"
                  r"newer|more recent|superseded", re.IGNORECASE)
CANT_TELL = re.compile(r"can'?t tell|cannot tell|can'?t determine|cannot determine|unclear which|"
                       r"not sure which|unknown|don'?t know which", re.IGNORECASE)


def real_chunk(source: dict, chunks: list[tuple[str, str, dict]]) -> SearchHit:
    for chunk_id, text, meta in chunks:
        if (meta["repo"] == source["repo"] and meta["path"] == source["path"]
                and source["title"].casefold() in meta["title"].casefold()):
            return SearchHit(chunk_id, 1.0, text, {**meta, "last_commit": source["date"]})
    raise LookupError(f"no chunk for {source}")


def main() -> None:
    cases = json.loads((ROOT / "data" / "golden_conflicts.json").read_text(encoding="utf-8"))
    payload = get_collection(DEFAULT_PERSIST_DIR, DEFAULT_MODEL).get(include=["documents", "metadatas"])
    chunks = list(zip(payload["ids"], payload["documents"], payload["metadatas"]))
    generator = ClaudeGenerator("claude-haiku-5-5", name="haiku")
    rows = []
    for number, case in enumerate(cases):
        real = real_chunk(case["source"], chunks)
        hits = [real]
        if case["copy"]:
            text = real.text
            pairs = case["copy"]["replace"]
            for old, new in zip(pairs[::2], pairs[1::2]):
                if old not in text:
                    raise ValueError(f"{case['id']}: {old!r} not in the source chunk")
                text = text.replace(old, new)
            copy = SearchHit(f"{real.chunk_id}-copy", 1.0, text,
                             {**real.meta, "path": case["copy"]["path"], "last_commit": case["copy"]["date"]})
            hits = [copy, real] if number % 2 else [real, copy]  # vary the order
        answer = generator.answer(case["question"], hits)
        text = answer.answer
        conflict = bool(case["copy"])
        rows.append({
            "id": case["id"], "kind": case["kind"], "question": case["question"],
            "answer": text, "answered": answer.answered,
            "flagged": bool(FLAG.search(text)) if conflict else None,
            "invented_conflict": bool(re.search(r"conflict|disagree|contradict|inconsisten", text, re.I))
            if not conflict else None,
            "has_current": (case["current"] in text) if case["current"] else None,
            "says_cant_tell": bool(CANT_TELL.search(text)) if case["current"] is None else None,
            "expected": case["why"],
        })
        print(f"{case['id']:11} {case['kind']:20} flagged={rows[-1]['flagged']} "
              f"current={rows[-1]['has_current']} cant_tell={rows[-1]['says_cant_tell']} "
              f"invented={rows[-1]['invented_conflict']}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    (OUT / f"{stamp}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet = ["# Conflicting sources: grade by hand", "",
             "Grade: flags the conflict (y/n), prefers the current source with the reason (y/n), or for "
             "controls: no invented conflict (y/n).", ""]
    for r in rows:
        sheet += [f"## {r['id']} ({r['kind']})", f"- **Question:** {r['question']}",
                  f"- **Expected:** {r['expected']}", f"- **Answer:** {r['answer']}",
                  f"- **Auto-check:** flagged={r['flagged']} current={r['has_current']} "
                  f"cant_tell={r['says_cant_tell']} invented={r['invented_conflict']}", "- **Grade:** ", ""]
    (OUT / f"grading_{stamp}.md").write_text("\n".join(sheet), encoding="utf-8")
    print(f"saved results/conflicts/{stamp}.json and grading_{stamp}.md")


if __name__ == "__main__":
    main()
