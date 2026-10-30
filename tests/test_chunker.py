from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from helpdesk.chunker import chunk_documents
from helpdesk.loader import Document, load_documents
from helpdesk.sources import REPOS


def words(text: str) -> int:
    return len(text.split())


def doc(text: str, path: str = "docs/x.md", kind: str = "doc") -> Document:
    return Document(
        text=textwrap.dedent(text).strip("\n"),
        repo="",
        path=path,
        kind=kind,
        last_commit="",
    )


def chunk(
    source: Document,
    max_tokens: int = 400,
    **kwargs: object,
):
    return chunk_documents([source], max_tokens, words, **kwargs)


def test_hash_comment_inside_fenced_block_is_not_a_heading() -> None:
    source = doc(
        """\
        # Setup
        Install dependencies first, then start the service.

        ```bash
        # install deps
        pip install -r requirements.txt
        ## not a heading either
        ```

        ~~~
        Done.
        ~~~
        """
    )

    chunks = chunk(source, tiny_tokens=0)

    assert [item.title for item in chunks] == ["Setup"]
    assert "# install deps" in chunks[0].text


def test_nested_heading_path_keeps_parent_context() -> None:
    source = doc(
        """\
        # Persistence
        How state survives restarts.
        ## Mongo handler
        Writes snapshots to Mongo.
        ### Retry policy
        Three retries with backoff.
        ## Redis cache
        Short-lived keys.
        """
    )

    titles = [item.title for item in chunk(source, tiny_tokens=0)]

    assert "Persistence > Mongo handler > Retry policy" in titles
    assert "Persistence > Redis cache" in titles


def test_heading_only_section_merges_into_next() -> None:
    source = doc(
        """\
        # Persistence
        ## Mongo handler
        Writes snapshots to Mongo with a write concern of majority, retried on failover.
        """
    )

    chunks = chunk(source, tiny_tokens=0)

    assert len(chunks) == 1
    assert chunks[0].title == "Persistence > Mongo handler"


def test_big_markdown_chunks_fit_budget_and_include_heading_context() -> None:
    paragraphs = "\n\n".join(
        " ".join(f"p{paragraph}_{word}" for word in range(25))
        for paragraph in range(12)
    )
    body = "\n".join(
        [
            "# Big",
            "## Many paragraphs",
            paragraphs,
            "## Long lines",
            "\n".join(" ".join(f"line{line}_{word}" for word in range(20)) for line in range(30)),
            "## Code",
            "```python\n" + "\n".join(f"cmd({index}) --flag value" for index in range(60)) + "\n```",
        ]
    )

    chunks = chunk(doc(body), max_tokens=100)

    assert len(chunks) > 5
    assert all(item.n_tokens == words(item.text) <= 100 for item in chunks)
    assert any(item.title == "Big > Long lines" for item in chunks)


def test_overlap_makes_neighboring_chunks_share_a_source_line() -> None:
    source = doc(
        "# Big\n## Long Lines\n"
        + "\n".join(f"line{index} has four words" for index in range(20))
    )

    chunks = [
        item
        for item in chunk(source, max_tokens=50, overlap_tokens=25, tiny_tokens=0)
        if item.title.startswith("Big > Long Lines")
    ]

    assert len(chunks) > 1
    assert any(
        right.line_start <= left.line_end
        for left, right in zip(chunks, chunks[1:])
    ), "overlapping chunks should share source lines"


def test_no_nonblank_markdown_line_is_lost() -> None:
    source = doc(
        "# Big\n"
        + "\n\n".join(
            f"paragraph{index} has words and content"
            for index in range(10)
        )
        + "\n## Code\n```python\n"
        + "\n".join(f"command({index})" for index in range(20))
        + "\n```"
    )
    chunks = chunk(source, max_tokens=10, tiny_tokens=0)
    chunk_lines = "\n".join(item.text for item in chunks)

    for line in source.text.split("\n"):
        if line.strip():
            assert line in chunk_lines, f"source line was lost: {line!r}"


def test_single_over_budget_line_splits_on_words() -> None:
    source = doc("# Huge\n" + " ".join(f"w{index}" for index in range(25)))

    chunks = chunk(source, max_tokens=10, tiny_tokens=0)

    assert len(chunks) > 1
    assert all(item.n_tokens <= 10 for item in chunks)
    assert " ".join(item.text for item in chunks).count("w24") == 1


SAMPLE_TEST = '''"""Tests for the pricer."""
import pytest

TIMEOUT = 5

@pytest.fixture
def pricer(client):
    return make_pricer(client)

def helper(x):
    return x * 2

def test_first(pricer):
    assert pricer.price() > 0

# Regression for a rounding bug
@pytest.mark.parametrize("value", [1, 2])
def test_second(pricer, value, tmp_path):
    assert helper(value)

async def test_third():
    assert True

class TestGroup:
    def test_in_class(self, pricer):
        assert pricer
'''

CONFTEST = '''import pytest

@pytest.fixture
def client():
    return object()
'''


def _line_of(text: str, needle: str) -> int:
    return next(index for index, line in enumerate(text.split("\n"), 1) if needle in line)


def _py_chunks(max_tokens: int = 400):
    test_doc = doc(SAMPLE_TEST, path="tests/unit/test_pricer.py", kind="test")
    conf_doc = doc(CONFTEST, path="tests/conftest.py", kind="test")
    chunks = chunk_documents([test_doc, conf_doc], max_tokens, words)
    return {item.title: item for item in chunks if item.path == test_doc.path}, chunks


def test_each_function_is_a_chunk_with_original_decorators_and_line_numbers() -> None:
    chunks, _ = _py_chunks()
    expected = {
        "test_first": (
            _line_of(SAMPLE_TEST, "def test_first"),
            _line_of(SAMPLE_TEST, "assert pricer.price()"),
        ),
        "test_second": (
            _line_of(SAMPLE_TEST, "# Regression"),
            _line_of(SAMPLE_TEST, "assert helper"),
        ),
        "test_third": (
            _line_of(SAMPLE_TEST, "async def test_third"),
            _line_of(SAMPLE_TEST, "assert True"),
        ),
    }

    for name, line_range in expected.items():
        assert (chunks[name].line_start, chunks[name].line_end) == line_range
    assert "TestGroup" in chunks


def test_module_header_holds_imports_docstring_and_constants() -> None:
    chunks, _ = _py_chunks()
    header = chunks["Module header"]

    assert "Tests for the pricer" in header.text
    assert "import pytest" in header.text
    assert "TIMEOUT = 5" in header.text
    assert "def test_first" not in header.text


def test_fixture_pointer_is_in_text_metadata_and_resolves_parent_conftest() -> None:
    chunks, all_chunks = _py_chunks()
    second = chunks["test_second"]

    assert second.text.split("\n")[0] == "# pytest fixtures: pricer, tmp_path"
    assert "pricer@tests/unit/test_pricer.py" in second.fixtures
    assert second.fixtures == ("pricer@tests/unit/test_pricer.py", "tmp_path")
    assert any(
        item.title == "pricer" and "client@tests/conftest.py" in item.fixtures
        for item in all_chunks
    )


def test_large_class_splits_per_method() -> None:
    methods = "\n".join(
        f"    def test_m{method}(self):\n"
        + "\n".join(
            f"        assert value_{method}_{index} == value_{method}_{index}"
            for index in range(6)
        )
        for method in range(6)
    )
    source = doc(f"class TestBig:\n{methods}", path="tests/test_big.py", kind="test")

    chunks = chunk(source, max_tokens=40)

    titles = [item.title for item in chunks]
    assert titles == [f"TestBig.test_m{index}" for index in range(6)]
    assert all(item.n_tokens <= 40 for item in chunks)


def test_syntax_error_falls_back_without_crashing() -> None:
    source = doc(
        "def test_broken(:\n    # not a heading\n    assert True\n\ndef test_ok():\n    pass\n",
        path="tests/test_broken.py",
        kind="test",
    )

    chunks = chunk(source)

    assert chunks
    text = "\n".join(item.text for item in chunks)
    assert "def test_broken(" in text and "def test_ok()" in text
    assert all(item.title.startswith("test_broken.py") for item in chunks)


def test_ids_are_stable_across_runs_edits_and_duplicate_titles() -> None:
    first = chunk(
        doc(
            "# API\n\n## Example\nfirst example text\n"
            "## Example\nsecond example text\n"
            "## Example\nthird example text\n"
        ),
        tiny_tokens=0,
    )
    edited = chunk(
        doc(
            "# API\n\n## Example\nA lengthened first example text\n"
            "## Example\nsecond example text\n"
            "## Example\nthird example text\n"
        ),
        tiny_tokens=0,
    )
    rerun = chunk(
        doc(
            "# API\n\n## Example\nA lengthened first example text\n"
            "## Example\nsecond example text\n"
            "## Example\nthird example text\n"
        ),
        tiny_tokens=0,
    )

    assert len(first) == len({item.id for item in first}) == 3
    assert [item.id for item in edited] == [item.id for item in rerun]
    assert [item.id for item in first] == [item.id for item in edited]


def test_real_chunk_ids_are_unique_across_available_repositories() -> None:
    if any(not path.is_dir() for path in REPOS.values()):
        pytest.skip("repositories are not available on this machine")
    docs = load_documents(REPOS)
    chunks = chunk_documents(docs, 400, words)
    ids = [item.id for item in chunks]

    assert len(ids) == len(set(ids))
