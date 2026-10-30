"""Hand a question off to a person: a suggested contact and a local ticket.

The contact is the most frequent author of recent non-merge commits to the
top candidate files (``git log``). It is a *suggested contact*, not an owner:
commit authorship isn't ownership. With no git, or files outside a git repo,
there is no suggestion and the ticket says so.

Tickets are markdown files in data/escalations/. Nothing is posted to Slack
or Jira; sending things outward is a separate decision that needs approval.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from helpdesk.index import SearchHit
from helpdesk.sources import REPOS

DEFAULT_ESCALATIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "escalations"
RECENT_COMMITS = 20


@dataclass(frozen=True)
class Contact:
    name: str
    commits: int
    path: str  # the file whose history named them


def recent_authors(repo_root: Path, path: str, n: int = RECENT_COMMITS) -> Counter[str]:
    """Authors of the last ``n`` non-merge commits touching ``path``; empty without git."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "log", "--no-merges", f"-n{n}", "--format=%an", "--", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Counter()
    if completed.returncode != 0:
        return Counter()
    return Counter(line.strip() for line in completed.stdout.splitlines() if line.strip())


def suggested_contact(
    files: list[tuple[str, str]],
    repos: dict[str, Path] = REPOS,
    n: int = RECENT_COMMITS,
) -> Contact | None:
    """Most active recent author over the candidate files, in order of relevance."""
    for repo, path in files:
        root = repos.get(repo)
        if root is None:
            continue
        authors = recent_authors(root, path, n)
        if authors:
            name, commits = authors.most_common(1)[0]
            return Contact(name, commits, f"{repo}/{path}")
    return None


def candidate_files(hits: list[SearchHit], limit: int = 5) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for hit in hits:
        key = (str(hit.meta.get("repo", "")), str(hit.meta.get("path", "")))
        if key not in files:
            files.append(key)
        if len(files) == limit:
            break
    return files


def write_ticket(
    *,
    question: str,
    condensed: str,
    summary: str,
    searched: str,
    hits: list[SearchHit],
    reason: str,
    contact: Contact | None,
    dir: Path = DEFAULT_ESCALATIONS_DIR,
) -> Path:
    directory = Path(dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{stamp}.md"
    files = candidate_files(hits)
    lines = [
        f"# Help desk escalation {stamp}",
        "",
        f"**Question:** {question}",
        f"**Condensed question:** {condensed}",
        f"**Why escalated:** {reason}",
        "",
        "## Conversation so far",
        summary or "(first turn)",
        "",
        "## What was searched",
        searched,
        "",
        "## Top candidate files",
        *(f"- {repo}/{file}" for repo, file in files),
        "",
        "## Suggested contact",
        (f"{contact.name} ({contact.commits} of the last {RECENT_COMMITS} non-merge commits to "
         f"{contact.path}). Suggested from git history, not an owner."
         if contact else "None found (no git history available for the candidate files)."),
        "",
        "---",
        "Drafted locally by the help desk; not sent anywhere.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def escalation_text(ticket: Path, contact: Contact | None, reason: str = "") -> str:
    if contact:
        who = f"for {contact.name}, who's been working on {contact.path} recently"
    else:
        who = "(I couldn't suggest a contact: there's no git history for those files)"
    if "asked for a person" in reason:
        opening = "Sure, I'll hand this to a person."
    elif "answer was wrong" in reason:
        opening = "Sorry about that. I'll pass this to a person to check."
    else:
        opening = "I couldn't find this in the docs."
    return f"{opening} I've drafted a hand-off {who}: {ticket}"
