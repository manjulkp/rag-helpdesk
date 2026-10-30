"""Print document counts and measure the loader's per-file Git calls.

Run with ``python -m scripts.load_stats`` from the project root.
"""

from collections import Counter
from time import perf_counter

import helpdesk.loader as loader
from helpdesk.sources import REPOS

git_calls = 0
git_seconds = 0.0
_original_last_commit = loader.last_commit


def _timed_last_commit(file_path):
    global git_calls, git_seconds
    started = perf_counter()
    try:
        return _original_last_commit(file_path)
    finally:
        git_calls += 1
        git_seconds += perf_counter() - started


def main() -> None:
    global git_calls, git_seconds
    git_calls = 0
    git_seconds = 0.0
    loader.last_commit = _timed_last_commit

    started = perf_counter()
    documents = loader.load_documents(REPOS)
    load_seconds = perf_counter() - started
    counts = Counter((document.repo, document.kind) for document in documents)

    print(f"{'repo':32} {'doc':>5} {'test':>5} {'total':>5}")
    for repo in REPOS:
        docs = counts[(repo, "doc")]
        tests = counts[(repo, "test")]
        print(f"{repo:32} {docs:5} {tests:5} {docs + tests:5}")

    total_docs = sum(counts[(repo, "doc")] for repo in REPOS)
    total_tests = sum(counts[(repo, "test")] for repo in REPOS)
    print(f"{'TOTAL':32} {total_docs:5} {total_tests:5} {len(documents):5}")
    untracked = sum(not document.last_commit for document in documents)
    print(f"Untracked (no last commit): {untracked}")
    print(f"Load time: {load_seconds:.2f}s")
    average_ms = git_seconds / max(git_calls, 1) * 1000
    print(
        f"Git calls: {git_calls} in {git_seconds:.2f}s "
        f"({average_ms:.1f}ms/call)"
    )


if __name__ == "__main__":
    main()
