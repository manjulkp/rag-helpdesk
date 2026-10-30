import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from helpdesk.loader import Document, load_documents
from helpdesk.sources import REPOS


def write_file(path: Path, text: str = "helpdesk content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_skips_virtual_environments_and_cache_directories(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_file(root / "docs" / "guide.md")
    write_file(root / ".venv" / "Lib" / "site-packages" / "ignored.md")
    write_file(root / "docs" / "__pycache__" / "ignored.md")

    documents = load_documents({"repo": root})

    assert [document.path for document in documents] == ["docs/guide.md"]
    assert all(".venv" not in document.path for document in documents)
    assert all("__pycache__" not in document.path for document in documents)


def test_loads_every_markdown_file_under_unity_platform_docs(tmp_path: Path) -> None:
    docs = tmp_path / "unity-platform" / "docs"
    expected = {
        write_file(docs / "start.md"),
        write_file(docs / "guides" / "nested" / "setup.md"),
    }
    write_file(docs / "guides" / "ignored.txt")

    documents = load_documents({"unity-platform": docs})

    actual = {docs / document.path for document in documents}
    assert actual == expected
    assert all(document.repo == "unity-platform" for document in documents)
    assert all(document.kind == "doc" for document in documents)


def test_last_commit_is_empty_or_iso_date(tmp_path: Path) -> None:
    file_path = write_file(tmp_path / "repo" / "docs" / "guide.md")

    documents = load_documents({"repo": file_path.parents[1]})

    assert len(documents) == 1
    assert documents[0].last_commit == "" or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", documents[0].last_commit
    )


def test_loads_python_only_under_tests_and_skips_empty_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_file(root / "README.md")
    write_file(root / "tests" / "test_feature.py", "def test_feature(): pass")
    write_file(root / "src" / "not_a_test.py", "print('not loaded')")
    write_file(root / "empty.md", "")
    write_file(root / "tests" / "empty_test.py", "   ")

    documents = load_documents({"repo": root})

    assert [(document.path, document.kind) for document in documents] == [
        ("README.md", "doc"),
        ("tests/test_feature.py", "test"),
    ]


def test_invalid_utf8_is_decoded_with_replacement(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    invalid_file = root / "guide.md"
    invalid_file.parent.mkdir(parents=True)
    invalid_file.write_bytes(b"valid text \xff")

    documents = load_documents({"repo": root})

    assert len(documents) == 1
    assert documents[0].text == "valid text \ufffd"


def test_missing_repo_is_skipped(tmp_path: Path) -> None:
    assert load_documents({"missing": tmp_path / "not-a-repo"}) == []


def test_result_has_document_fields(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_file(root / "guide.md")

    document = load_documents({"repo": root})[0]

    assert isinstance(document, Document)
    assert document.text == "helpdesk content"
    assert document.repo == "repo"
    assert document.path == "guide.md"
    assert document.kind == "doc"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Loader Test",
            "GIT_AUTHOR_EMAIL": "loader@example.test",
            "GIT_COMMITTER_NAME": "Loader Test",
            "GIT_COMMITTER_EMAIL": "loader@example.test",
        },
    )


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("Git is required for repository metadata tests")

    repo = tmp_path / "toy"
    files = {
        "README.md": "# Toy repository\n",
        "docs/deep/nested/guide.md": "nested documentation\n",
        "tests/test_a.py": "def test_a(): pass\n",
        "tests/integration/test_b.py": "def test_b(): pass\n",
        "tests/notes.md": "test notes\n",
        "src/not_a_test.py": "print('not loaded')\n",
        "tests/venv/lib/site.py": "ignored virtualenv file\n",
        "tests/__pycache__/junk.py": "ignored bytecode cache file\n",
    }
    for relative_path, text in files.items():
        write_file(repo / relative_path, text)
    latin1_file = repo / "docs" / "latin1.md"
    latin1_file.write_bytes("café menu\n".encode("latin-1"))
    (repo / "docs" / "binary.md").write_bytes(b"\x00\x01")
    (repo / "empty.md").write_bytes(b"")

    _git(repo, "init", "-q")
    _git(repo, "add", "README.md", "tests/test_a.py")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_toy_repo_selection_paths_and_kinds(toy_repo: Path) -> None:
    documents = load_documents({"toy": toy_repo})
    by_path = {document.path: document for document in documents}

    assert set(by_path) == {
        "README.md",
        "docs/deep/nested/guide.md",
        "docs/latin1.md",
        "tests/test_a.py",
        "tests/integration/test_b.py",
        "tests/notes.md",
    }
    assert by_path["tests/integration/test_b.py"].kind == "test"
    assert by_path["tests/notes.md"].kind == "doc"
    # non-UTF-8 files are decoded with replacement characters, not guessed as
    # Latin-1 (see test_invalid_utf8_is_decoded_with_replacement); this assertion
    # only ran once git was installed, and contradicted that test
    assert by_path["docs/latin1.md"].text == "caf� menu\n"


def test_committed_and_untracked_files_have_expected_last_commit(toy_repo: Path) -> None:
    documents = {doc.path: doc for doc in load_documents({"toy": toy_repo})}

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", documents["README.md"].last_commit)
    assert documents["tests/test_a.py"].last_commit == documents["README.md"].last_commit
    assert documents["docs/deep/nested/guide.md"].last_commit == ""


def test_subdirectory_scan_keeps_paths_relative_to_repository(toy_repo: Path) -> None:
    documents = load_documents({"toy": toy_repo / "docs"})

    assert {document.path for document in documents} == {
        "docs/deep/nested/guide.md",
        "docs/latin1.md",
    }


@pytest.fixture(scope="module")
def real_docs() -> list[Document]:
    missing = [name for name, path in REPOS.items() if not path.is_dir()]
    if missing:
        pytest.skip(f"repositories not available on this machine: {', '.join(missing)}")
    return load_documents(REPOS)


def test_real_repo_documents_exclude_skipped_directories(real_docs: list[Document]) -> None:
    skipped = {".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    for document in real_docs:
        assert not skipped.intersection(Path(document.path).parts)
        assert not any(part.endswith(".egg-info") for part in Path(document.path).parts)


def test_research_desk_sources_are_docs(real_docs: list[Document]) -> None:
    research_docs = [
        document
        for document in real_docs
        if document.repo == "research-desk" and document.kind == "doc"
    ]

    assert research_docs, "expected documents from research-desk"
    assert any(document.path == "README.md" for document in research_docs)


def test_real_repo_last_commits_are_empty_or_iso_dates(real_docs: list[Document]) -> None:
    assert all(
        document.last_commit == ""
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", document.last_commit)
        for document in real_docs
    )
