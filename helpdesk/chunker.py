"""Split documents into token-budgeted chunks for embedding and retrieval.

Markdown is split by heading, while pytest files are split by AST node. Large
sections fall back to paragraph, line, and finally word boundaries. Every
chunk retains the source path and one-based inclusive line range.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

from helpdesk.loader import Document

CountTokens = Callable[[str], int]

# BGE-small-v1.5 accepts 512 tokens. Reserve space for special tokens,
# contextual descriptions, and tokenizer variance.
DEFAULT_MAX_TOKENS = 400
LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    repo: str
    path: str
    kind: str
    title: str
    line_start: int
    line_end: int
    last_commit: str
    n_tokens: int
    fixtures: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Piece:
    title: str
    line_start: int
    line_end: int
    text: str
    fixtures: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Section:
    title: str
    lo: int
    hi: int
    has_heading: bool


def chunk_documents(
    docs: list[Document],
    max_tokens: int,
    count_tokens: CountTokens,
    overlap_tokens: int | None = None,
    tiny_tokens: int | None = None,
) -> list[Chunk]:
    """Chunk documents while keeping chunks within the supplied token budget.

    ``max_tokens`` defaults are intentionally left to callers so a later model
    change only requires a new token counter and budget. Tiny sections are
    folded into neighbors; overlap is measured with the same counter.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    overlap = max_tokens // 10 if overlap_tokens is None else overlap_tokens
    tiny = max_tokens // 16 if tiny_tokens is None else tiny_tokens
    if overlap < 0 or tiny < 0:
        raise ValueError("overlap_tokens and tiny_tokens must not be negative")

    splitter = _Splitter(count_tokens, overlap, tiny)
    fixtures = _FixtureIndex(docs)
    chunks: list[Chunk] = []
    for doc in docs:
        if not doc.text.strip():
            continue
        if doc.kind == "test":
            tree = fixtures.tree(doc)
            pieces = (
                _chunk_plain(doc, splitter, max_tokens)
                if tree is None
                else _chunk_python(doc, tree, splitter, fixtures, max_tokens)
            )
        else:
            pieces = _chunk_markdown(doc, splitter, max_tokens)
        chunks.extend(_finalise(doc, pieces, count_tokens))
    return chunks


def _finalise(
    doc: Document, pieces: list[_Piece], count_tokens: CountTokens
) -> list[Chunk]:
    occurrences: Counter[str] = Counter()
    chunks: list[Chunk] = []
    for piece in pieces:
        occurrence = occurrences[piece.title]
        occurrences[piece.title] += 1
        chunks.append(
            Chunk(
                id=chunk_id(doc.repo, doc.path, piece.title, occurrence),
                text=piece.text,
                repo=doc.repo,
                path=doc.path,
                kind=doc.kind,
                title=piece.title,
                line_start=piece.line_start,
                line_end=piece.line_end,
                last_commit=doc.last_commit,
                n_tokens=count_tokens(piece.text),
                fixtures=piece.fixtures,
            )
        )
    return chunks


def chunk_id(repo: str, path: str, title: str, occurrence: int) -> str:
    """Make an ID from location and title, not chunk text or line numbers."""
    key = f"{repo}\0{path}\0{title}\0{occurrence}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _split_lines(text: str) -> list[str]:
    return LINE_BREAK.split(text)


def _with_parts(
    title: str,
    parts: list[tuple[int, int, str]],
    line_offset: int,
) -> list[_Piece]:
    return [
        _Piece(title, line_offset + lo, line_offset + hi - 1, text)
        for lo, hi, text in parts
        if text.strip()
    ]


def _fence_open(line: str) -> tuple[str, int] | None:
    match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
    if not match:
        return None
    marker, info = match.groups()
    if marker[0] == "`" and "`" in info:
        return None
    return marker[0], len(marker)


def _fence_close(line: str, fence: tuple[str, int]) -> bool:
    marker, minimum = fence
    match = re.match(r"^ {0,3}(`+|~+)[ \t]*$", line)
    return bool(match and match.group(1)[0] == marker and len(match.group(1)) >= minimum)


def _atx_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^ {0,3}(#{1,3})(?:[ \t]+|$)(.*)$", line)
    if not match:
        return None
    title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
    return len(match.group(1)), title


def _heading_text(title: str) -> str:
    return re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()


def _md_sections(lines: list[str], intro_title: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[str] = []
    title = intro_title
    start = 0
    fence: tuple[str, int] | None = None

    for index, line in enumerate(lines):
        if fence is not None:
            if _fence_close(line, fence):
                fence = None
            continue
        opened = _fence_open(line)
        if opened is not None:
            fence = opened
            continue

        heading = _atx_heading(line)
        if heading is None:
            continue
        if start < index:
            sections.append(_Section(title, start, index, bool(stack)))
        level, heading_title = heading
        stack = stack[: level - 1]
        if heading_title:
            stack.append(heading_title)
        title = " > ".join(stack) if stack else intro_title
        start = index

    if start < len(lines):
        sections.append(_Section(title, start, len(lines), bool(stack)))
    return sections or [_Section(intro_title, 0, len(lines), False)]


def _merge_tiny(
    sections: list[_Section],
    lines: list[str],
    count_tokens: CountTokens,
    tiny_tokens: int,
) -> list[_Section]:
    """Fold small heading-only sections forward; fold a final one backward."""
    result: list[_Section] = []
    pending: _Section | None = None
    for section in sections:
        body_start = section.lo + (1 if section.has_heading else 0)
        body = "\n".join(lines[body_start : section.hi]).strip()
        if pending is not None:
            section = _Section(
                section.title, pending.lo, section.hi, section.has_heading
            )
            pending = None
        if count_tokens(body) <= tiny_tokens and section is not sections[-1]:
            pending = section
        else:
            result.append(section)
    if pending is not None:
        if result:
            previous = result[-1]
            result[-1] = _Section(
                previous.title, previous.lo, pending.hi, previous.has_heading
            )
        else:
            result.append(pending)
    return result


def _chunk_markdown(doc: Document, splitter: _Splitter, max_tokens: int) -> list[_Piece]:
    lines = _split_lines(doc.text)
    intro = PurePosixPath(doc.path).name
    sections = _md_sections(lines, intro)
    sections = _merge_tiny(
        sections, lines, splitter.count, splitter.tiny_tokens
    )
    pieces: list[_Piece] = []
    for section in sections:
        split = splitter.split(lines[section.lo : section.hi], section.lo + 1, max_tokens)
        pieces.extend(_with_parts(section.title, split, 0))
    return pieces


def _chunk_plain(doc: Document, splitter: _Splitter, max_tokens: int) -> list[_Piece]:
    lines = _split_lines(doc.text)
    title = PurePosixPath(doc.path).name
    return _with_parts(
        title,
        splitter.split(lines, 1, max_tokens),
        0,
    )


def _decorator_target(decorator: ast.expr) -> ast.expr:
    return decorator.func if isinstance(decorator, ast.Call) else decorator


def _is_fixture(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = _decorator_target(decorator)
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
    return False


def _fixture_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = _decorator_target(decorator)
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "fixture"
            or isinstance(target, ast.Name)
            and target.id == "fixture"
        ):
            for keyword in decorator.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    if isinstance(keyword.value.value, str):
                        return keyword.value.value
    return node.name


class _FixtureIndex:
    """Resolve pytest fixture definitions within the same repository."""

    def __init__(self, docs: list[Document]) -> None:
        self._trees: dict[tuple[str, str], ast.Module | None] = {}
        self._definitions: dict[tuple[str, str], set[str]] = {}
        for doc in docs:
            if doc.kind != "test":
                continue
            key = (doc.repo, doc.path)
            try:
                tree = ast.parse(doc.text)
            except SyntaxError:
                tree = None
            self._trees[key] = tree
            if tree is None:
                continue
            names = {
                _fixture_name(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _is_fixture(node)
            }
            self._definitions[key] = names

    def tree(self, doc: Document) -> ast.Module | None:
        return self._trees.get((doc.repo, doc.path))

    def resolve(self, doc: Document, name: str) -> str:
        path = PurePosixPath(doc.path)
        parent = path.parent
        candidates = [doc.path]
        while True:
            conftest = (parent / "conftest.py").as_posix()
            if conftest not in candidates:
                candidates.append(conftest)
            if parent == PurePosixPath("."):
                break
            parent = parent.parent
        for candidate in candidates:
            if name in self._definitions.get((doc.repo, candidate), set()):
                return f"{name}@{candidate}"
        return name


def _requested_fixtures(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    parametrized: set[str] = set()
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = _decorator_target(decorator)
        if not isinstance(target, ast.Attribute) or target.attr != "parametrize":
            continue
        if decorator.args and isinstance(decorator.args[0], ast.Constant):
            value = decorator.args[0].value
            if isinstance(value, str):
                parametrized.update(name.strip() for name in value.split(","))
            elif isinstance(value, (tuple, list)):
                parametrized.update(
                    arg.value for arg in value if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                )

    names = [
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"} | parametrized
    ]
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = _decorator_target(decorator)
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == "usefixtures"
        ):
            continue
        names.extend(
            arg.value
            for arg in decorator.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        )
    return list(dict.fromkeys(names))


def _node_start(node: ast.stmt, lines: list[str] | None = None) -> int:
    decorators = getattr(node, "decorator_list", ())
    start = min([node.lineno, *(decorator.lineno for decorator in decorators)])
    if lines is None:
        return start
    index = start - 2
    while index >= 0 and lines[index].lstrip().startswith("#"):
        start = index + 1
        index -= 1
    return start


def _node_kind(node: ast.stmt) -> str:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if _is_fixture(node):
            return "fixture"
        return "test" if node.name.startswith("test") else "helper"
    return "header"


def _code_pieces(
    title: str,
    lines: list[str],
    lo: int,
    hi: int,
    fixtures: tuple[str, ...],
    splitter: _Splitter,
    max_tokens: int,
) -> list[_Piece]:
    text_lines = lines[lo - 1 : hi]
    prefix = _fixture_line(fixtures, splitter.count, max_tokens // 4)
    budget = max_tokens - splitter.count(prefix) if prefix else max_tokens
    if budget < 1:
        raise ValueError("max_tokens leaves no room for code after fixture context")
    pieces = splitter.split(text_lines, lo, budget)
    out = _with_parts(title, pieces, 0)
    if prefix:
        out = [
            _Piece(
                piece.title,
                piece.line_start,
                piece.line_end,
                f"{prefix}\n{piece.text}",
                fixtures,
            )
            for piece in out
        ]
    else:
        out = [
            _Piece(
                piece.title,
                piece.line_start,
                piece.line_end,
                piece.text,
                fixtures,
            )
            for piece in out
        ]
    return out


def _fixture_line(
    fixtures: tuple[str, ...], count_tokens: CountTokens, limit: int
) -> str:
    names: list[str] = []
    for fixture in fixtures:
        display_name = fixture.split("@", 1)[0]
        candidate = ", ".join((*names, display_name))
        line = f"# pytest fixtures: {candidate}"
        if count_tokens(line) > limit:
            break
        names.append(display_name)
    return f"# pytest fixtures: {', '.join(names)}" if names else ""


def _chunk_python(
    doc: Document,
    tree: ast.Module,
    splitter: _Splitter,
    fixtures: _FixtureIndex,
    max_tokens: int,
) -> list[_Piece]:
    lines = _split_lines(doc.text)
    nodes = list(tree.body)
    pieces: list[_Piece] = []
    cursor = 1
    for node in nodes:
        is_chunk_node = isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        if not is_chunk_node:
            continue
        start = _node_start(node, lines)
        end = getattr(node, "end_lineno", node.lineno)
        if cursor < start:
            pieces.extend(
                _code_pieces(
                    "Module header",
                    lines,
                    cursor,
                    start - 1,
                    (),
                    splitter,
                    max_tokens,
                )
            )
        kind = _node_kind(node)
        if kind == "class" and isinstance(node, ast.ClassDef):
            pieces.extend(
                _chunk_class(doc, node, lines, start, end, splitter, fixtures, max_tokens)
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            requested = _requested_fixtures(node)
            resolved = tuple(
                fixtures.resolve(doc, name) for name in requested
            )
            title = _fixture_name(node) if kind == "fixture" else node.name
            pieces.extend(
                _code_pieces(title, lines, start, end, resolved, splitter, max_tokens)
            )
        cursor = max(cursor, end + 1)
    if cursor <= len(lines):
        pieces.extend(
            _code_pieces(
                "Module header",
                lines,
                cursor,
                len(lines),
                (),
                splitter,
                max_tokens,
            )
        )
    return pieces


def _chunk_class(
    doc: Document,
    cls: ast.ClassDef,
    lines: list[str],
    lo: int,
    hi: int,
    splitter: _Splitter,
    fixtures: _FixtureIndex,
    max_tokens: int,
) -> list[_Piece]:
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if splitter.count("\n".join(lines[lo - 1 : hi])) <= max_tokens:
        return _code_pieces(cls.name, lines, lo, hi, (), splitter, max_tokens)
    if not methods:
        return _code_pieces(cls.name, lines, lo, hi, (), splitter, max_tokens)

    out: list[_Piece] = []
    previous_end = lo - 1
    for method_index, method in enumerate(methods):
        method_start = _node_start(method, lines)
        method_end = method.end_lineno or method.lineno
        piece_start = lo if method_index == 0 else max(previous_end + 1, method_start)
        requested = _requested_fixtures(method)
        resolved = tuple(fixtures.resolve(doc, name) for name in requested)
        out.extend(
            _code_pieces(
                f"{cls.name}.{method.name}",
                lines,
                piece_start,
                method_end,
                resolved,
                splitter,
                max_tokens,
            )
        )
        previous_end = method_end
    if previous_end < hi and methods:
        out.extend(
            _code_pieces(
                f"{cls.name}.{methods[-1].name}",
                lines,
                previous_end + 1,
                hi,
                (),
                splitter,
                max_tokens,
            )
        )
    return out


def _paragraphs(lines: list[str]) -> list[tuple[int, int]]:
    """Return non-blank paragraph spans, keeping fenced blocks as one span."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    fence: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        if fence is not None:
            if _fence_close(line, fence):
                fence = None
            continue
        opened = _fence_open(line)
        if opened is not None:
            if start is None:
                start = index
            fence = opened
            continue
        if line.strip():
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(lines)))
    return spans


class _Splitter:
    def __init__(
        self,
        count_tokens: CountTokens,
        overlap_tokens: int,
        tiny_tokens: int,
    ) -> None:
        self.count = count_tokens
        self.overlap_tokens = overlap_tokens
        self.tiny_tokens = tiny_tokens

    def split(
        self, lines: list[str], first_line: int, budget: int
    ) -> list[tuple[int, int, str]]:
        atoms: list[tuple[int, int, str]] = []
        for lo, hi in _paragraphs(lines):
            text = "\n".join(lines[lo:hi]).strip()
            if self.count(text) <= budget:
                atoms.append((lo, hi, text))
                continue
            for line_index in range(lo, hi):
                line = lines[line_index]
                if not line.strip():
                    continue
                if self.count(line) <= budget:
                    atoms.append((line_index, line_index + 1, line))
                    continue
                atoms.extend(
                    (line_index, line_index + 1, piece)
                    for piece in self._hard_split(line, budget)
                )

        output: list[tuple[int, int, str]] = []
        current: list[tuple[int, int, str]] = []
        current_tokens = 0
        for atom in atoms:
            atom_tokens = self.count(atom[2])
            separator_tokens = self.count("\n") if current else 0
            if current and current_tokens + separator_tokens + atom_tokens > budget:
                output.append(self._join(current))
                overlap = self._overlap(current, first_line)
                current = overlap + [atom]
                current_tokens = self.count("\n".join(part[2] for part in current))
                while len(current) > 1 and current_tokens > budget:
                    current.pop(0)
                    current_tokens = self.count("\n".join(part[2] for part in current))
            else:
                current.append(atom)
                current_tokens = self.count("\n".join(part[2] for part in current))

        if current:
            output.append(self._join(current))

        checked: list[tuple[int, int, str]] = []
        for lo, hi, text in output:
            if self.count(text) <= budget:
                checked.append((first_line + lo, first_line + hi, text))
            else:
                for piece in self._hard_split(text, budget):
                    checked.append((first_line + lo, first_line + hi, piece))
        return checked

    @staticmethod
    def _join(atoms: list[tuple[int, int, str]]) -> tuple[int, int, str]:
        return atoms[0][0], atoms[-1][1], "\n".join(atom[2] for atom in atoms)

    def _overlap(
        self, atoms: list[tuple[int, int, str]], first_line: int
    ) -> list[tuple[int, int, str]]:
        del first_line
        selected: list[tuple[int, int, str]] = []
        tokens = 0
        for atom in reversed(atoms):
            cost = self.count(atom[2])
            if selected and tokens + cost > self.overlap_tokens:
                break
            if cost > self.overlap_tokens:
                break
            selected.insert(0, atom)
            tokens += cost
        return selected

    def _hard_split(self, text: str, budget: int) -> list[str]:
        words = text.split()
        pieces: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if self.count(candidate) <= budget:
                current = candidate
                continue
            if current:
                pieces.append(current)
                current = ""
            if self.count(word) <= budget:
                current = word
                continue
            fragment = ""
            for character in word:
                candidate = fragment + character
                if self.count(candidate) <= budget:
                    fragment = candidate
                else:
                    if not fragment:
                        raise ValueError("max_tokens is too small for the token counter")
                    pieces.append(fragment)
                    fragment = character
            if fragment:
                current = fragment
        if current:
            pieces.append(current)
        return pieces or [text]
