from __future__ import annotations

import re
import typing as t
from bisect import bisect_left
from dataclasses import dataclass
from difflib import SequenceMatcher

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import Dialect, DialectType
    from sqlglot.errors import ErrorLevel
    from sqlglot.expressions.core import Expr
    from sqlglot.tokens import Token

# Segment kinds
COPY: t.Final[str] = "copy"
"""An output fragment copied verbatim (modulo casing) from the input."""

MERGE: t.Final[str] = "merge"
"""An output fragment that replaces an input fragment at the same position (e.g. a rename)."""

SYNTHETIC: t.Final[str] = "synthetic"
"""A newly invented output fragment with no corresponding input fragment."""

FORMAT: t.Final[str] = "format"
"""Generated whitespace, indentation or line breaks introduced by formatting."""

COMMENT: t.Final[str] = "comment"
"""A comment; it carries its source range when it could be located in the input."""

_SENTINEL = "__SQLGLOT__LB__"

# Quoted strings/identifiers are one indivisible atom: the whitespace and
# punctuation they contain is part of the literal, not formatting or syntax.
# This also covers T-SQL's bracket-delimited identifiers.
_QUOTED_RE: t.Final[str] = (
    r"""'(?:\\.|''|[^'\\])*'|"(?:\\.|""|[^"\\])*"|`(?:``|[^`])*`|\[(?:[^\]]|\]\])*\]"""
)

# A line break sentinel, a run of whitespace, a quoted literal, a word/number or a
# single punctuation character. Input tokens and output gaps are scanned with the
# same atom definition.
_GAP_RE: t.Final[re.Pattern[str]] = re.compile(
    re.escape(_SENTINEL) + r"|\s+|" + _QUOTED_RE + r"|\w+|[^\w\s]", re.UNICODE
)
_ATOM_RE: t.Final[re.Pattern[str]] = re.compile(
    _QUOTED_RE + r"|\w+|[^\w\s]", re.UNICODE
)
_COMMENT_RE: t.Final[re.Pattern[str]] = re.compile(r"/\*.*?\*/", re.DOTALL)
# When locating a rendered child piece inside its parent's text, occurrences inside
# quoted literals or comments must be ignored (e.g. the `b` inside `/* block */`).
_PIECE_SKIP_RE: t.Final[re.Pattern[str]] = re.compile(
    _QUOTED_RE + r"|/\*.*?\*/|--[^\n]*|#(?=[ \t])[^\n]*", re.DOTALL
)


@dataclass(frozen=True)
class Segment:
    """
    A half-open character interval (`start`, `end`) of the generated SQL.

    `source_start` / `source_end` are the corresponding half-open character interval in
    the original input string, or `None` when the fragment has no source.
    """

    start: int
    end: int
    source_start: int | None
    source_end: int | None
    kind: str

    @property
    def source(self) -> tuple[int, int] | None:
        if self.source_start is None or self.source_end is None:
            return None
        return self.source_start, self.source_end


@dataclass
class SourceMap:
    """
    The source map of a single generated SQL statement.

    `segments` tile `sql` completely: every output character belongs to exactly one
    segment, and every segment either points at an exact interval of the input or is
    explicitly marked sourceless. `error` is set when the statement could not be mapped;
    in that case the other statements are still mapped and the input is left untouched.
    """

    sql: str
    source: str
    segments: list[Segment]
    error: str | None = None

    def origin(self, offset: int) -> tuple[int, int] | None:
        """Returns the input interval the output character at `offset` was derived from."""
        for segment in self.segments:
            if segment.start <= offset < segment.end:
                return segment.source
        return None

    def segments_for(self, start: int, end: int) -> list[Segment]:
        """Returns all segments overlapping the output interval [`start`, `end`)."""
        return [
            segment
            for segment in self.segments
            if segment.start < end and segment.end > start
        ]

    def origins_for(self, start: int, end: int) -> list[tuple[int, int]]:
        """Returns the distinct, ordered input intervals covering the output interval."""
        origins: list[tuple[int, int]] = []
        for segment in self.segments_for(start, end):
            source = segment.source
            if source and (not origins or origins[-1] != source):
                origins.append(source)
        return origins

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "sql": self.sql,
            "error": self.error,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "source_start": segment.source_start,
                    "source_end": segment.source_end,
                    "kind": segment.kind,
                }
                for segment in self.segments
            ],
        }


class _Frame:
    __slots__ = ("node", "text", "pieces", "children", "_final")

    def __init__(self, node: Expr):
        self.node = node
        self.text = ""
        # Child frames rendered while generating the node, in call order.
        self.pieces: list[_Frame] = []
        # (offset within text, child frame, core start within the child piece, core length)
        self.children: list[tuple[int, _Frame, int, int]] = []
        self._final: list[Segment] | None = None


def _find_outside(text: str, core: str, cursor: int) -> int:
    """Finds `core` in `text` at/after `cursor`, skipping quoted/comment regions."""
    position = cursor
    while True:
        index = text.find(core, position)
        if index < 0:
            return -1
        end = index + len(core)
        for match in _PIECE_SKIP_RE.finditer(text, cursor):
            skip_start, skip_end = match.span()
            if skip_end <= index:
                continue
            if skip_start < end and skip_end > index:
                position = skip_end
                break
        else:
            return index
        if position <= index:
            position = index + 1


def _find_piece(text: str, piece: str, cursor: int) -> tuple[int, int, int] | None:
    """Finds `piece` in `text` after `cursor`.

    Returns (piece start in text, piece core start in piece, core length).
    """
    index = _find_outside(text, piece, cursor)
    if index >= 0:
        return index, 0, len(piece)

    # A few generators post-process the rendered child (e.g. `.strip()`); locate its
    # non-whitespace core instead.
    lead = len(piece) - len(piece.lstrip())
    core = piece.strip()
    if core:
        index = _find_outside(text, core, cursor)
        if index >= 0:
            return index, lead, len(core)
    return None


def _make_mapping_sql() -> t.Callable[..., str]:
    from sqlglot.generator import Generator

    def mapping_sql(
        self: t.Any,
        expression: str | Expr | None,
        key: str | None = None,
        comment: bool = True,
    ) -> str:
        if not expression:
            return ""

        if isinstance(expression, str):
            # Plain strings emitted by a generator (keywords, function names, parser
            # string args, ...) are not placed: the gaps around the children are aligned
            # with the input structural tokens and unmatched strings become sourceless.
            return expression

        if key:
            value = expression.args.get(key)
            if value:
                return self.sql(value)
            return ""

        frame = _Frame(expression)
        stack = self._map_stack
        stack.append(frame)
        try:
            text = Generator.sql(self, expression, key=None, comment=comment)
        except BaseException:
            stack.pop()
            raise
        stack.pop()

        frame.text = text
        cursor = 0
        for child in frame.pieces:
            piece = child.text
            if not piece:
                continue
            located = _find_piece(text, piece, cursor)
            if located is None:
                continue
            index, lead, core_len = located
            # Only the located core of the child is placed; its own segments are
            # restricted to the core and the surrounding whitespace becomes a gap.
            frame.children.append((index, child, lead, core_len))
            cursor = index + core_len

        if stack:
            stack[-1].pieces.append(frame)
        else:
            self._map_roots.append(frame)
        return text

    return mapping_sql


_MAPPING_GENERATOR_CLASSES: dict[type, type] = {}


def _iter_slots(obj: t.Any) -> t.Iterator[tuple[str, t.Any]]:
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name in ("__dict__", "__weakref__"):
                continue
            try:
                yield name, getattr(obj, name)
            except AttributeError:
                pass


def _instrument_delegates(owner: t.Any, stack: list[_Frame], roots: list[_Frame]) -> None:
    """Shares the tracing state with generators an owner delegates to.

    Some dialects (e.g. Athena) generate by forwarding to a different generator
    instance; those instances must push onto the same frame stack.
    """
    from sqlglot.generator import Generator

    for attr, value in list(_iter_slots(owner)):
        if not isinstance(value, Generator) or getattr(value, "_map_stack", None) is stack:
            continue
        mapping_cls = _mapping_generator_class(type(value))
        replacement = mapping_cls.__new__(mapping_cls)
        for name, state in _iter_slots(value):
            setattr(replacement, name, state)
        replacement._map_stack = stack
        replacement._map_roots = roots
        setattr(owner, attr, replacement)
        _instrument_delegates(replacement, stack, roots)


def _mapping_generator_class(generator_class: type) -> type:
    cls = _MAPPING_GENERATOR_CLASSES.get(generator_class)
    if cls is None:

        def __init__(self: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
            generator_class.__init__(self, *args, **kwargs)
            self._map_stack: list[_Frame] = []
            self._map_roots: list[_Frame] = []
            _instrument_delegates(self, self._map_stack, self._map_roots)

        cls = type(
            f"Mapping{generator_class.__name__}",
            (generator_class,),
            {
                "__slots__": ("_map_stack", "_map_roots"),
                "__init__": __init__,
                "sql": _make_mapping_sql(),
            },
        )
        _MAPPING_GENERATOR_CLASSES[generator_class] = cls
    return cls


def _seg(start: int, end: int, source: tuple[int, int] | None, kind: str) -> Segment:
    return Segment(start, end, source[0] if source else None, source[1] if source else None, kind)


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda s: (s.start, s.end)):
        last = merged[-1] if merged else None
        if (
            last is not None
            and segment.start == last.end
            and segment.kind == last.kind
            and segment.source_start == last.source_start
            and segment.source_end == last.source_end
        ):
            merged[-1] = Segment(
                last.start,
                segment.end,
                last.source_start,
                last.source_end,
                last.kind,
            )
        else:
            merged.append(segment)
    return merged


def _leading_keywords(token_type: t.Any) -> dict[str, frozenset[t.Any]]:
    """Maps an unanchored node class to the clause keywords that lead it."""
    tt = token_type
    join_modifiers = frozenset(
        filter(
            None,
            (
                getattr(tt, name, None)
                for name in (
                    "JOIN",
                    "STRAIGHT_JOIN",
                    "APPLY",
                    "INNER",
                    "OUTER",
                    "CROSS",
                    "LEFT",
                    "RIGHT",
                    "FULL",
                    "NATURAL",
                    "SEMI",
                    "ANTI",
                )
            ),
        )
    )
    keywords: dict[str, frozenset[t.Any]] = {
        "Select": frozenset(filter(None, (getattr(tt, "SELECT", None),))),
        "Distinct": frozenset(filter(None, (getattr(tt, "DISTINCT", None),))),
        "From": frozenset(filter(None, (getattr(tt, "FROM", None),))),
        "Where": frozenset(filter(None, (getattr(tt, "WHERE", None),))),
        "Group": frozenset(filter(None, (getattr(tt, "GROUP_BY", None),))),
        "Having": frozenset(filter(None, (getattr(tt, "HAVING", None),))),
        "Order": frozenset(filter(None, (getattr(tt, "ORDER_BY", None),))),
        "Limit": frozenset(filter(None, (getattr(tt, "LIMIT", None),))),
        "Offset": frozenset(filter(None, (getattr(tt, "OFFSET", None),))),
        "Qualify": frozenset(filter(None, (getattr(tt, "QUALIFY", None),))),
        "Join": join_modifiers,
        "Union": frozenset(filter(None, (getattr(tt, "UNION", None),))),
        "Intersect": frozenset(filter(None, (getattr(tt, "INTERSECT", None),))),
        "Except": frozenset(filter(None, (getattr(tt, "EXCEPT", None),))),
        "With": frozenset(
            filter(None, (getattr(tt, "WITH", None), getattr(tt, "RECURSIVE", None)))
        ),
        "Values": frozenset(filter(None, (getattr(tt, "VALUES", None),))),
        "Insert": frozenset(
            filter(None, (getattr(tt, "INSERT", None), getattr(tt, "INTO", None)))
        ),
        "Update": frozenset(filter(None, (getattr(tt, "UPDATE", None),))),
        "Delete": frozenset(filter(None, (getattr(tt, "DELETE", None),))),
        "Merge": frozenset(filter(None, (getattr(tt, "MERGE", None),))),
        "Set": frozenset(filter(None, (getattr(tt, "SET", None),))),
        "Create": frozenset(
            filter(
                None,
                (getattr(tt, "CREATE", None), getattr(tt, "OR", None), getattr(tt, "REPLACE", None)),
            )
        ),
        "Alter": frozenset(filter(None, (getattr(tt, "ALTER", None),))),
        "Drop": frozenset(filter(None, (getattr(tt, "DROP", None),))),
        "Grant": frozenset(filter(None, (getattr(tt, "GRANT", None),))),
        "Revoke": frozenset(filter(None, (getattr(tt, "REVOKE", None),))),
        "Window": frozenset(filter(None, (getattr(tt, "WINDOW", None),))),
        "Pivot": frozenset(
            filter(None, (getattr(tt, "PIVOT", None), getattr(tt, "UNPIVOT", None)))
        ),
        "Connect": frozenset(filter(None, (getattr(tt, "CONNECT_BY", None),))),
        "Lateral": frozenset(
            filter(None, (getattr(tt, "LATERAL", None), getattr(tt, "VIEW", None)))
        ),
        "Returning": frozenset(filter(None, (getattr(tt, "RETURNING", None),))),
        "And": frozenset(filter(None, (getattr(tt, "AND", None),))),
        "Or": frozenset(filter(None, (getattr(tt, "OR", None),))),
        "Not": frozenset(filter(None, (getattr(tt, "NOT", None),))),
        "Is": frozenset(filter(None, (getattr(tt, "IS", None),))),
        "In": frozenset(filter(None, (getattr(tt, "IN", None),))),
        "Like": frozenset(
            filter(None, (getattr(tt, "LIKE", None), getattr(tt, "ILIKE", None)))
        ),
        "Between": frozenset(filter(None, (getattr(tt, "BETWEEN", None),))),
        "Exists": frozenset(filter(None, (getattr(tt, "EXISTS", None),))),
        "Case": frozenset(filter(None, (getattr(tt, "CASE", None),))),
        "When": frozenset(
            filter(None, (getattr(tt, "WHEN", None), getattr(tt, "THEN", None)))
        ),
        "Else": frozenset(filter(None, (getattr(tt, "ELSE", None),))),
    }
    return keywords


def _compute_spans(
    root: Expr, content_tokens: list[Token]
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, tuple[int, int]],
    dict[int, int],
]:
    """Computes the source token spans (inclusive ends) of every parsed node.

    Returns the tight spans (anchors and descendants only) and the extended spans.
    Node meta only anchors the token at which a node starts parsing. Nodes created
    without an anchor token (e.g. `Select`, `From`) would therefore miss their leading
    keywords (`SELECT`, `FROM`, ...). Their extended span reaches left over their own
    leading keywords (e.g. `TRY_CAST(` for a `Cast`, `FROM` for a `From`) but never
    across clause keywords belonging to an enclosing or sibling construct. Every span
    also reaches right through closing brackets opened inside it (e.g. the `)` of
    `CAST(...)` or `COUNT(*)`). Extended spans may overlap; overlapping tokens are
    attributed to the frame that actually renders them.
    """
    from sqlglot.tokens import TokenType

    nodes = list(root.dfs())

    seed_spans: dict[int, tuple[int, int]] = {}
    anchored: set[int] = set()
    for node in reversed(nodes):
        start = end = None
        meta = node._meta
        if meta is not None and meta.get("start") is not None:
            start, end = meta["start"], meta["end"]
            anchored.add(id(node))

        for child in node.iter_expressions():
            child_span = seed_spans.get(id(child))
            if child_span is not None:
                child_start, child_end = child_span
                start = child_start if start is None else min(start, child_start)
                end = child_end if end is None else max(end, child_end)

        if start is not None:
            seed_spans[id(node)] = (start, end)

    def is_descendant(maybe_descendant: Expr, ancestor: Expr) -> bool:
        parent = maybe_descendant.parent
        while parent is not None:
            if parent is ancestor:
                return True
            parent = parent.parent
        return False

    open_types = {TokenType.L_PAREN, TokenType.L_BRACKET, TokenType.L_BRACE}
    close_types = {TokenType.R_PAREN, TokenType.R_BRACKET, TokenType.R_BRACE}
    leading = _leading_keywords(TokenType)
    keyword_leaders: dict[t.Any, set[str]] = {}
    for class_name, types in leading.items():
        for token_type in types:
            keyword_leaders.setdefault(token_type, set()).add(class_name)

    ext_spans = dict(seed_spans)
    right_caps: dict[int, int] = {}
    for node in nodes:
        seed = seed_spans.get(id(node))
        if seed is None:
            continue

        if id(node) in anchored:
            left = seed[0]
        else:
            # Never reach into a preceding sibling construct or past an enclosing
            # node's own anchor.
            hard = 0
            boundary = -1
            for other in nodes:
                if other is node:
                    continue
                other_span = seed_spans.get(id(other))
                if other_span is not None and other_span[1] < seed[0]:
                    boundary = max(boundary, other_span[1])

            parent = node.parent
            while parent is not None:
                parent_span = seed_spans.get(id(parent))
                if (
                    parent_span is not None
                    and id(parent) in anchored
                    and parent_span[0] < seed[0]
                ):
                    boundary = max(boundary, parent_span[0])
                parent = parent.parent
            hard = boundary + 1

            # Walk the tokens left of the seed, picking up the node's own leading
            # syntax: an optional call prefix (`FUNC(`), clause keywords led by this
            # node's class, and arbitrary keywords that no ancestor construct leads
            # (e.g. `OVERWRITE TABLE` for `Insert`). Everything else (operators,
            # commas, closing brackets, keywords led by an ancestor such as `SELECT`
            # or `AND`) ends the walk.
            own = leading.get(type(node).__name__, frozenset())
            ancestor_classes = set()
            ancestor = node.parent
            while ancestor is not None:
                ancestor_classes.add(type(ancestor).__name__)
                ancestor = ancestor.parent

            left = seed[0]
            index = len(content_tokens) - 1
            while index >= 0:
                token = content_tokens[index]
                if token.end >= seed[0] or token.start < hard:
                    index -= 1
                    continue
                token_type = token.token_type
                if token_type in close_types:
                    break
                if token_type in open_types:
                    # Claim the bracket only as the node's own call prefix when a
                    # word is glued to it (e.g. `TRY_CAST(`); a bracket belonging to
                    # an enclosing construct stays with that frame.
                    previous = content_tokens[index - 1] if index > 0 else None
                    if (
                        previous is not None
                        and previous.end + 1 == token.start
                        and previous.start >= hard
                        and (
                            previous.text[:1].isalnum() or previous.text[:1] == "_"
                        )
                        and keyword_leaders.get(previous.token_type, set()).isdisjoint(
                            ancestor_classes
                        )
                    ):
                        left = previous.start
                        index -= 2
                        continue
                    break
                leaders = keyword_leaders.get(token_type)
                if leaders is not None:
                    if token_type in own or leaders.isdisjoint(ancestor_classes):
                        left = token.start
                        index -= 1
                        continue
                    break
                first = token.text[:1]
                if first.isalnum() or first == "_" or first in "'\"`[":
                    # Identifier/word belonging to this node's own prefix.
                    left = token.start
                    index -= 1
                    continue
                break

        # Hard stop: a sibling construct starting past the seed's right end.
        right_cap = 10**18
        for other in nodes:
            if other is node or is_descendant(other, node):
                continue
            other_span = seed_spans.get(id(other))
            if other_span is not None and other_span[0] > seed[1]:
                right_cap = min(right_cap, other_span[0])

        balance = 0
        right = seed[1]
        for token in content_tokens:
            if token.start >= left and token.end <= seed[1]:
                if token.token_type in open_types:
                    balance += 1
                elif token.token_type in close_types and balance > 0:
                    balance -= 1
            elif token.start > seed[1] and token.start < right_cap and balance > 0:
                right = max(right, token.end)
                if token.token_type in open_types:
                    balance += 1
                elif token.token_type in close_types:
                    balance -= 1
                    if balance <= 0:
                        break

        ext_spans[id(node)] = (left, right)
        right_caps[id(node)] = right_cap

    return seed_spans, ext_spans, right_caps


def _deepest_rendered(
    root: Expr,
    token_start: int,
    token_end: int,
    spans: dict[int, tuple[int, int]],
    rendered: set[int],
) -> Expr | None:
    """Finds the deepest rendered node whose source span contains the given token."""
    root_span = spans.get(id(root))
    if root_span is None or not (root_span[0] <= token_start and token_end <= root_span[1]):
        return None

    node = root
    chain = [root]
    while True:
        descended = False
        for child in node.iter_expressions():
            child_span = spans.get(id(child))
            if child_span and child_span[0] <= token_start and token_end <= child_span[1]:
                node = child
                chain.append(child)
                descended = True
                break
        if not descended:
            break

    for node in reversed(chain):
        if id(node) in rendered:
            return node
    return None


def _scan_input_token(token: Token, source_sql: str) -> list[tuple[int, int, str]]:
    """Splits an input token's raw source slice into (half-open offsets, normalized atom text).

    The raw source is scanned rather than `token.text`: multi-word keywords (e.g.
    `GROUP  BY`) and string tokens normalize or strip characters in `.text`, which
    would desynchronize the offsets.
    """
    return [
        (
            token.start + match.start(),
            token.start + match.end(),
            match.group() if match.group()[0] in "'\"`[" else match.group().upper(),
        )
        for match in _ATOM_RE.finditer(source_sql[token.start : token.end + 1])
    ]


def _comment_content(raw: str) -> str:
    """Normalizes a comment to its payload so that `-- x` matches `/* x */`."""
    text = raw.strip()
    if text.startswith("/*") and text.endswith("*/"):
        return text[2:-2].strip()
    if text.startswith("--"):
        return text[2:].strip()
    if text.startswith("#"):
        return text[1:].strip()
    return text


def _inside_quoted_text(text: str, start: int, position: int) -> bool:
    """Best-effort check that `position` lies within a quoted or bracketed literal."""
    quote = None
    index = start
    while index < position:
        char = text[index]
        if quote:
            if quote == "[":
                if text[index : index + 2] == "]]":
                    index += 1
                elif char == "]":
                    quote = None
            elif char == "\\":
                index += 1
            elif text[index : index + 2] == quote * 2:
                index += 1
            elif char == quote:
                quote = None
        elif char in "'\"`[":
            quote = char
        index += 1
    return quote is not None


class _Mapper:
    def __init__(
        self,
        root: _Frame,
        input_comments: list[tuple[int, int, str]],
        global_atoms: list[tuple[int, int, str]],
        tight_owners: dict[int, int],
        tight_atoms: dict[int, list[int]],
        ext_spans: dict[int, tuple[int, int]],
        right_caps: dict[int, int],
    ):
        self.root = root
        self.input_comments = input_comments
        self.global_atoms = global_atoms
        # global atom index -> owning node id
        self.owners = dict(tight_owners)
        self.ext_spans = ext_spans
        self.right_caps = right_caps
        self.owned_atoms: dict[int, list[int]] = {
            node_id: list(atom_ids) for node_id, atom_ids in tight_atoms.items()
        }

        # Per-frame gap analysis: output atoms (offset within frame, norm) and the
        # non-comment gap chunks that are eligible for alignment.
        self.frame_output_atoms: dict[int, list[tuple[int, int, str]]] = {}
        self.frame_chunks: dict[int, list[tuple[int, int]]] = {}
        self.frame_comments: dict[int, list[tuple[int, int, tuple[int, int] | None]]] = {}
        self.frame_gap_segments: dict[int, list[Segment]] = {}

        self.frames = list(self._iter_preorder(root))
        for frame in reversed(self.frames):
            self._analyze_frame(frame)
        self._assign_comments()
        self._allocate()

    def _iter_preorder(self, frame: _Frame) -> t.Iterator[_Frame]:
        stack = [frame]
        while stack:
            current = stack.pop()
            yield current
            for _, child, _, _ in reversed(current.children):
                stack.append(child)

    @staticmethod
    def _is_ancestor_node(ancestor_id: int | None, node: Expr) -> bool:
        parent = node.parent
        while parent is not None:
            if id(parent) == ancestor_id:
                return True
            parent = parent.parent
        return False

    def _analyze_frame(self, frame: _Frame) -> None:
        text = frame.text
        covered = [(offset, offset + core_len) for offset, _, _, core_len in frame.children]
        gaps = _complement(covered, len(text))

        comment_spans: list[tuple[int, int, tuple[int, int] | None]] = []
        chunks: list[tuple[int, int]] = []
        output_atoms: list[tuple[int, int, str]] = []

        for gap_start, gap_end in gaps:
            spans: list[tuple[int, int]] = []
            for match in _COMMENT_RE.finditer(text, gap_start, gap_end):
                if _inside_quoted_text(text, gap_start, match.start()):
                    continue
                spans.append((match.start(), match.end()))
            comment_spans.extend((start, end, None) for start, end in spans)

            for chunk_start, chunk_end in _complement(spans, gap_end, gap_start):
                chunks.append((chunk_start, chunk_end))
                for token_match in _GAP_RE.finditer(
                    text, chunk_start, chunk_end
                ):
                    token = token_match.group()
                    if not token[0].isspace() and token != _SENTINEL:
                        output_atoms.append(
                            (
                                token_match.start(),
                                token_match.end(),
                                token if token[0] in "'\"`[" else token.upper(),
                            )
                        )

        key = id(frame)
        self.frame_output_atoms[key] = output_atoms
        self.frame_chunks[key] = chunks
        self.frame_comments[key] = comment_spans

    def _collect_comments(
        self,
        frame: _Frame,
        zero: int,
        lo: int,
        hi: int,
        found: list[tuple[int, int, int]],
    ) -> None:
        # Absolute offset of frame coordinate x is `zero + x - lo`.
        for index, (start, end, _) in enumerate(self.frame_comments[id(frame)]):
            if start >= lo and end <= hi:
                found.append((zero + start - lo, id(frame), index))
        for offset, child, lead, core_len in frame.children:
            self._collect_comments(
                child, zero + offset - lo - lead, lead, lead + core_len, found
            )

    def _assign_comments(self) -> None:
        # Pair rendered comments with input comments in output order, matching by
        # normalized payload (the generator rewrites `-- x` into `/* x */`) and
        # falling back to source order.
        rendered: list[tuple[int, int, int]] = []
        self._collect_comments(self.root, 0, 0, len(self.root.text), rendered)
        rendered.sort(key=lambda item: item[0])

        used = [False] * len(self.input_comments)
        cursor = 0
        for _, frame_id, index in rendered:
            source: tuple[int, int] | None = None

            start, end, _ = self.frame_comments[frame_id][index]
            frame = next(frame for frame in self.frames if id(frame) == frame_id)
            payload = _comment_content(frame.text[start:end])

            picked = next(
                (
                    j
                    for j in range(cursor, len(self.input_comments))
                    if not used[j] and self.input_comments[j][2] == payload
                ),
                None,
            )
            if picked is None and cursor < len(self.input_comments):
                picked = next(
                    (j for j in range(cursor, len(self.input_comments)) if not used[j]),
                    None,
                )
            if picked is not None:
                used[picked] = True
                cursor = picked + 1
                source = (self.input_comments[picked][0], self.input_comments[picked][1])

            self.frame_comments[frame_id][index] = (start, end, source)

    def _allocate(self) -> None:
        # Children before parents: tight structures claim first, parent frames then
        # align their own keywords against input tokens that are still unclaimed.
        for frame in reversed(self.frames):
            node_id = id(frame.node)
            ext_span = self.ext_spans.get(node_id)
            output_atoms = self.frame_output_atoms[id(frame)]
            gap_segments: list[Segment] = []

            for chunk_start, chunk_end in self.frame_chunks[id(frame)]:
                for token_match in _GAP_RE.finditer(
                    frame.text, chunk_start, chunk_end
                ):
                    token = token_match.group()
                    if token[0].isspace() or token == _SENTINEL:
                        gap_segments.append(
                            _seg(token_match.start(), token_match.end(), None, FORMAT)
                        )

            if not output_atoms or ext_span is None:
                self.frame_gap_segments[id(frame)] = gap_segments
                continue

            # Unanchored trailing syntax (e.g. the `IS NULL` rendered by an `Is`
            # frame) may sit past the own extended span; atoms reserved by an
            # ancestor are reachable up to the next sibling construct.
            cap = self.right_caps.get(node_id, ext_span[1])
            candidate_indexes = []
            for index, atom in enumerate(self.global_atoms):
                owner = self.owners.get(index)
                ancestor_owned = owner is not None and owner != node_id and (
                    self._is_ancestor_node(owner, frame.node)
                )
                right_bound = cap if ancestor_owned else ext_span[1]
                if (
                    (owner is None or owner == node_id or ancestor_owned)
                    and ext_span[0] <= atom[0]
                    and atom[1] - 1 <= right_bound
                ):
                    candidate_indexes.append(index)

            if candidate_indexes:
                matcher = SequenceMatcher(
                    None,
                    [self.global_atoms[index][2] for index in candidate_indexes],
                    [atom[2] for atom in output_atoms],
                    autojunk=False,
                )
                chunks = self.frame_chunks[id(frame)]
                claimed = self.owned_atoms.setdefault(node_id, [])

                def emit(start: int, end: int, source: tuple[int, int] | None, kind: str) -> None:
                    # Restrict the range to gap chunks: it may geometrically span a
                    # covered child core or a comment sitting between its atoms.
                    for chunk_start, chunk_end in chunks:
                        clip_start = max(start, chunk_start)
                        clip_end = min(end, chunk_end)
                        if clip_end > clip_start:
                            gap_segments.append(
                                Segment(
                                    clip_start,
                                    clip_end,
                                    source[0] if source else None,
                                    source[1] if source else None,
                                    kind,
                                )
                            )

                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal":
                        for input_index, output_atom in zip(
                            candidate_indexes[i1:i2], output_atoms[j1:j2]
                        ):
                            input_atom = self.global_atoms[input_index]
                            emit(
                                output_atom[0],
                                output_atom[1],
                                (input_atom[0], input_atom[1]),
                                COPY,
                            )
                            owner = self.owners.get(input_index)
                            if owner != node_id:
                                self.owners[input_index] = node_id
                                claimed.append(input_index)
                    elif tag == "replace":
                        # Only atoms reserved for this frame itself are consumed;
                        # ancestor-owned atoms are matcher context that another frame
                        # (or the ancestor gap) still has to render.
                        consumed = [
                            input_index
                            for input_index in candidate_indexes[i1:i2]
                            if self.owners.get(input_index) in (None, node_id)
                        ]
                        for input_index in consumed:
                            self.owners[input_index] = node_id
                            claimed.append(input_index)
                        if consumed:
                            source_first = self.global_atoms[consumed[0]]
                            source_last = self.global_atoms[consumed[-1]]
                            source_range = (source_first[0], source_last[1])
                            kind = MERGE
                        else:
                            # The block only aligned against ancestor-owned context:
                            # the output is freshly generated syntax.
                            source_range = None
                            kind = SYNTHETIC
                        for output_atom in output_atoms[j1:j2]:
                            emit(
                                output_atom[0],
                                output_atom[1],
                                source_range,
                                kind,
                            )
                    elif tag == "insert":
                        for output_atom in output_atoms[j1:j2]:
                            emit(output_atom[0], output_atom[1], None, SYNTHETIC)
                    # "delete" drops input tokens without producing output.
                claimed.sort()
            else:
                for output_atom in output_atoms:
                    gap_segments.append(
                        Segment(
                            output_atom[0],
                            output_atom[1],
                            None,
                            None,
                            SYNTHETIC,
                        )
                    )

            self.frame_gap_segments[id(frame)] = gap_segments

    def finalize(self) -> list[Segment]:
        return self._finalize(self.root)

    def _finalize(self, frame: _Frame) -> list[Segment]:
        if frame._final is not None:
            return frame._final

        segments: list[Segment] = []

        for offset, child, lead, core_len in frame.children:
            child_segments = self._finalize(child)
            core_start = lead
            core_end = lead + core_len
            for segment in child_segments:
                if segment.end <= core_start or segment.start >= core_end:
                    continue
                start = max(segment.start, core_start) - core_start + offset
                end = min(segment.end, core_end) - core_start + offset
                if end > start:
                    segments.append(
                        Segment(
                            start,
                            end,
                            segment.source_start,
                            segment.source_end,
                            segment.kind,
                        )
                    )

        segments.extend(self.frame_gap_segments[id(frame)])
        for start, end, source in self.frame_comments[id(frame)]:
            segments.append(_seg(start, end, source, COMMENT))

        segments = _merge_adjacent(segments)
        frame._final = segments
        return segments


def _complement(
    intervals: list[tuple[int, int]],
    end: int,
    start: int = 0,
) -> list[tuple[int, int]]:
    complement: list[tuple[int, int]] = []
    cursor = start
    for interval_start, interval_end in sorted(intervals):
        if interval_start > end:
            break
        interval_start = max(interval_start, start)
        if interval_start > cursor:
            complement.append((cursor, min(interval_start, end)))
        cursor = max(cursor, min(interval_end, end))
    if cursor < end:
        complement.append((cursor, end))
    return complement


def _raw_offset_to_final(offset: int, lead: int, collapses: list[int]) -> int:
    """Maps an offset in the raw generated text to one in the stripped, line-broken text."""
    return offset - lead - 14 * bisect_left(collapses, offset)


def _map_statement(
    source_sql: str,
    expression: Expr,
    write_dialect: Dialect,
    tokens: list[Token],
    pretty: bool,
    opts: dict[str, t.Any],
) -> SourceMap:
    from sqlglot.tokens import TokenType

    generator_class = _mapping_generator_class(write_dialect.generator_class)
    generator = generator_class(dialect=write_dialect, **opts)

    output_sql = generator.generate(expression, copy=True)

    root_frame = generator._map_roots[-1]
    raw = root_frame.text

    content_tokens = [
        token for token in tokens if token.token_type != TokenType.SEMICOLON
    ]
    seed_spans, ext_spans, right_caps = _compute_spans(root_frame.node, content_tokens)

    # Extend the root span to the end of its statement chunk so that trailing tokens
    # owned by no descendant (e.g. unparsed keywords) remain attributable to the root.
    if content_tokens:
        root_id = id(root_frame.node)
        chunk_end = max(token.end for token in content_tokens)
        root_seed = seed_spans.get(root_id)
        root_ext = ext_spans.get(root_id)
        if root_ext is not None:
            ext_spans[root_id] = (root_ext[0], max(root_ext[1], chunk_end))
            if root_seed is not None:
                seed_spans[root_id] = (
                    root_seed[0],
                    max(root_seed[1], chunk_end),
                )
        else:
            first = content_tokens[0]
            ext_spans[root_id] = (first.start, chunk_end)

    # Comments of the statement chunk, in source order. Their attachment to parser
    # nodes is irrelevant: the generator relocates and sometimes rewrites them.
    input_comments: list[tuple[int, int, str]] = []
    seen_positions: set[tuple[int, int]] = set()
    for token in tokens:
        for position in token.comment_positions:
            if position is None or position in seen_positions:
                continue
            seen_positions.add(position)
            comment_start, comment_end = position
            raw_comment = source_sql[comment_start : comment_end + 1]
            input_comments.append(
                (comment_start, comment_end + 1, _comment_content(raw_comment))
            )
    input_comments.sort(key=lambda comment: comment[0])

    # Frames without a source span are transparent: their region is aligned by the
    # nearest placed ancestor and their placed children are adopted by it.
    _flatten_spanless_frames(root_frame, set(ext_spans))

    rendered = set()
    for frame in _iter_frames(root_frame):
        rendered.add(id(frame.node))

    # Tight ownership: tokens inside the tight span belong to the deepest rendered node.
    global_atoms: list[tuple[int, int, str]] = []
    tight_owners: dict[int, int] = {}
    tight_atoms: dict[int, list[int]] = {}
    for token in content_tokens:
        owner = _deepest_rendered(
            root_frame.node, token.start, token.end, seed_spans, rendered
        )
        for atom in _scan_input_token(token, source_sql):
            index = len(global_atoms)
            global_atoms.append(atom)
            if owner is not None:
                tight_owners[index] = id(owner)
                tight_atoms.setdefault(id(owner), []).append(index)

    raw_segments = _Mapper(
        root_frame,
        input_comments,
        global_atoms,
        tight_owners,
        tight_atoms,
        ext_spans,
        right_caps,
    ).finalize()

    # Mirror Generator.generate: strip, then collapse the pretty line break sentinels.
    lead = len(raw) - len(raw.lstrip())
    stripped_end = len(raw.rstrip())
    core = raw[lead:stripped_end]
    collapses = [
        lead + match.start()
        for match in re.finditer(re.escape(_SENTINEL), core)
    ]
    final = core.replace(_SENTINEL, "\n") if pretty else core

    segments: list[Segment] = []
    for segment in raw_segments:
        start = _raw_offset_to_final(segment.start, lead, collapses)
        end = _raw_offset_to_final(segment.end, lead, collapses)
        start = max(start, 0)
        end = min(end, len(final))
        if end > start:
            segments.append(
                Segment(
                    start,
                    end,
                    segment.source_start,
                    segment.source_end,
                    segment.kind,
                )
            )

    segments = _merge_adjacent(segments)

    map_ = SourceMap(sql=final, source=source_sql, segments=segments)
    if final != output_sql:
        # The traced output must be identical to the ordinary generator output.
        map_.error = "internal source map mismatch"
    return map_


def _iter_frames(root: _Frame) -> t.Iterator[_Frame]:
    stack = [root]
    while stack:
        frame = stack.pop()
        yield frame
        for _, child, _, _ in frame.children:
            stack.append(child)


def _flatten_spanless_frames(root: _Frame, placed: set[int]) -> None:
    """Absorbs frames without a source span into their parent.

    A frame without a span renders a parser- or generator-built node that has no
    input anchor (e.g. a bare `INT` data type or a synthesized `INT64` rewrite); it
    cannot own input atoms. Its rendered region becomes part of the parent gap and
    its placed children are adopted by the parent, so the parent's alignment can
    attribute the region (renames included).
    """

    def flatten(frame: _Frame) -> bool:
        adopted: list[tuple[int, _Frame, int, int]] = []
        kept: list[tuple[int, _Frame, int, int]] = []
        for offset, child, lead, core_len in frame.children:
            if flatten(child):
                kept.append((offset, child, lead, core_len))
                continue
            for child_offset, grandchild, grandchild_lead, grandchild_len in child.children:
                new_offset = offset + child_offset - lead
                if 0 <= new_offset and new_offset + grandchild_len <= len(frame.text):
                    adopted.append((new_offset, grandchild, grandchild_lead, grandchild_len))
        frame.children = sorted(kept + adopted, key=lambda item: item[0])
        return id(frame.node) in placed

    flatten(root)


def source_map(
    sql: str,
    read: DialectType = None,
    write: DialectType = None,
    identity: bool = True,
    error_level: ErrorLevel | None = None,
    **opts: t.Any,
) -> list[SourceMap]:
    """
    Transpiles `sql` between dialects and returns a `SourceMap` per generated statement.

    Each map ties every character interval of the generated SQL back to the exact
    character interval it came from in `sql`. Fragments produced by formatting,
    comments or generation (copied, merged/renamed or newly invented) are all marked,
    with sourceless fragments carrying an explicit `None` source.

    The mapping is deterministic, does not mutate `sql` or the parsed syntax trees, and
    isolates failures: a statement that cannot be mapped yields a `SourceMap` with its
    `error` set while the remaining statements are still mapped.

    Args:
        sql: the source SQL code.
        read: the source dialect.
        write: the target dialect (defaults to `read`).
        identity: use `read` as both source and target when `write` is not specified.
        error_level: the desired parser error level.
        **opts: other `sqlglot.generator.Generator` options (e.g. `pretty=True`).
    """
    from sqlglot.dialects.dialect import Dialect

    write = (read if write is None else write) if identity else write
    read_dialect = Dialect.get_or_raise(read)
    write_dialect = Dialect.get_or_raise(write)

    pretty = bool(opts.get("pretty", False))
    parse_opts: dict[str, t.Any] = {}
    if error_level is not None:
        parse_opts["error_level"] = error_level

    expressions = read_dialect.parse(sql, **parse_opts)
    # Re-tokenize on a fresh token stream so the parser's comment relocation cannot
    # affect the positions used for mapping, then split the tokens into statement
    # chunks in the same order as the parsed expressions.
    from sqlglot.tokens import TokenType

    chunks: list[list[Token]] = [[]]
    for token in read_dialect.tokenize(sql):
        chunks[-1].append(token)
        if token.token_type == TokenType.SEMICOLON:
            chunks.append([])
    if chunks and not chunks[-1]:
        chunks.pop()

    maps: list[SourceMap] = []
    for index, expression in enumerate(expressions):
        if expression is None:
            maps.append(SourceMap(sql="", source=sql, segments=[]))
            continue
        chunk = chunks[index] if index < len(chunks) else [
            token for chunk in chunks for token in chunk
        ]
        try:
            maps.append(
                _map_statement(sql, expression, write_dialect, chunk, pretty, opts)
            )
        except Exception as exception:  # failures must not affect other statements
            maps.append(
                SourceMap(sql="", source=sql, segments=[], error=str(exception))
            )
    return maps
