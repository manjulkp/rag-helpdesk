"""Who may see which repos.

``data/acl.json`` maps users to groups and groups to repos ("*" = all). The
filter is enforced where chunks enter the system, as a Chroma/BM25
``{"repo": {"$in": [...]}}`` filter on every search, and again on every other
path a chunk can take into a reply (sticky sources, the reranker's candidates,
the gate, clarification options, hand-off files): ``filter_hits`` in
pipeline.respond. Two layers, so a missed ``where`` can't leak.

A forbidden repo looks exactly like a missing one: the user gets "I couldn't
find this", never "you don't have access", because confirming that a
document exists is itself a leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helpdesk.index import SearchHit

DEFAULT_ACL = Path(__file__).resolve().parents[1] / "data" / "acl.json"


@dataclass(frozen=True)
class User:
    name: str
    repos: frozenset[str] | None  # None = every repo

    def may_see(self, repo: str) -> bool:
        return self.repos is None or repo in self.repos


def load_acl(path: str | Path = DEFAULT_ACL) -> dict[str, User]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups: dict[str, Any] = payload.get("groups", {})
    users: dict[str, User] = {}
    for name, spec in payload.get("users", {}).items():
        repos: set[str] = set()
        everything = False
        for group in spec.get("groups", []):
            if group not in groups:
                raise ValueError(f"user {name!r} is in unknown group {group!r}")
            members = groups[group]
            if members == "*":
                everything = True
            else:
                repos.update(members)
        users[name] = User(name, None if everything else frozenset(repos))
    return users


def repo_filter(user: User | None, where: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The search ``where`` for a user, ANDed with any existing filter."""
    if user is None or user.repos is None:
        return where
    clause = {"repo": {"$in": sorted(user.repos)}}
    if not where:
        return clause
    existing = where["$and"] if "$and" in where else [{k: v} for k, v in where.items()]
    return {"$and": [*existing, clause]}


def filter_hits(user: User | None, hits: list[SearchHit]) -> list[SearchHit]:
    if user is None:
        return hits
    return [hit for hit in hits if user.may_see(str(hit.meta.get("repo", "")))]
