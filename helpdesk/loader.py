from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class Document:
    text: str
    repo: str
    path: str
    kind: str
    last_commit: str


def load_documents(repos: dict[str, Path]) -> list[Document]:
    """Load Markdown and test Python files from the supplied repository paths."""
    documents: list[Document] = []
    for repo, scan_dir in repos.items():
        scan_dir = Path(scan_dir).expanduser().resolve()
        if not scan_dir.is_dir():
            log.warning("Skipping %s: %s is not a directory", repo, scan_dir)
            continue

        repo_root = _git_root(scan_dir) or scan_dir
        for file_path, kind in _candidate_files(scan_dir):
            text = _read_text(file_path)
            if text is None or not text.strip():
                continue
            try:
                relative_path = file_path.relative_to(repo_root).as_posix()
            except ValueError:
                relative_path = file_path.relative_to(scan_dir).as_posix()
            documents.append(
                Document(
                    text=text,
                    repo=repo,
                    path=relative_path,
                    kind=kind,
                    last_commit=last_commit(file_path),
                )
            )
    return documents


def _candidate_files(scan_dir: Path):
    """Yield Markdown files and Python files under the scan root's tests directory."""
    for directory, dirnames, filenames in os.walk(scan_dir):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIP_DIRS and not name.endswith(".egg-info")
        )
        relative_parts = directory_path.relative_to(scan_dir).parts
        in_tests = bool(relative_parts) and relative_parts[0] == "tests"
        for name in sorted(filenames):
            file_path = directory_path / name
            if name.endswith(".md"):
                yield file_path, "doc"
            elif name.endswith(".py") and in_tests:
                yield file_path, "test"


def _read_text(path: Path) -> str | None:
    """Read a text file; warn and skip unreadable, binary, or invalid files."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        log.warning("Skipping unreadable file %s: %s", path, error)
        return None

    if not raw or b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.warning("Decoding %s with replacement characters; it is not valid UTF-8", path)
        return raw.decode("utf-8-sig", errors="replace")


def _git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        log.warning("Unable to locate Git repository for %s: %s", path, error)
        return None
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 else None


def last_commit(file_path: Path) -> str:
    """Return the file's last commit date, or an empty string if it is untracked."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(file_path.parent),
                "log",
                "-1",
                "--format=%cs",
                "--",
                file_path.name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        log.warning("Unable to read Git history for %s: %s", file_path, error)
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()
