"""Omit source-price quotations without interpreting any JSON scalar values.

The private original is the authoritative audit artifact. Public ``null`` values
introduced here mean an omitted quotation, not a missing source observation.
This module performs no file reads or writes; callers must preserve originals
before replacing a public report.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re


POLICY_VERSION = "source-price-excerpts-v1"
REPLACEMENT_MEANING = (
    "Introduced nulls omit source-price quotations; they do not indicate "
    "missing values in the original source or audit."
)
_METADATA_KEY = "public_sanitization"
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_HEX = frozenset(b"0123456789abcdefABCDEF")

# Asterisks match one object key or array index. ``$series`` matches the root
# series key specifically, so removing a row's price never removes its parent.
_POLICIES = {
    "price_audit.json": (
        ("series", "*", "minimum", "value"),
        ("series", "*", "maximum", "value"),
        ("series", "*", "nonpositive_observations", "*", "value"),
        ("series", "*", "largest_absolute_adjacent_changes", "*", "previous_price"),
        ("series", "*", "largest_absolute_adjacent_changes", "*", "price"),
        ("series", "*", "largest_absolute_adjacent_changes", "*", "change_usd"),
        ("series", "*", "longest_calendar_gap", "previous_price"),
        ("series", "*", "longest_calendar_gap", "price"),
        ("series", "*", "longest_calendar_gap", "change_usd"),
    ),
    "vintage_audit.json": (
        ("*", "date_level_reconstruction_verified_against_samples", "*", "latest_value"),
        ("*", "first_numeric_lag_after_initial_vintage", "longest_examples", "*", "$series"),
        ("*", "same_or_future_observation_date_records", "*", "$series"),
        ("*", "impossible_future_observation_date_records", "*", "$series"),
        ("*", "event_examples", "*", "before"),
        ("*", "event_examples", "*", "after"),
    ),
    "source_reconciliation.json": (
        ("current_eia_vs_alfred", "*", "mismatches", "*", "eia"),
        ("current_eia_vs_alfred", "*", "mismatches", "*", "alfred"),
        ("sampled_weekly_releases", "*", "mismatches", "*", "archived_price"),
        ("sampled_weekly_releases", "*", "mismatches", "*", "current_price"),
    ),
}


@dataclass(frozen=True)
class _Node:
    start: int
    end: int
    kind: str
    fields: tuple = ()
    items: tuple = ()


class _Parser:
    """Validate JSON structure, decoding object keys and no scalar values."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def fail(self, message: str):
        raise ValueError(f"Invalid report JSON at byte {self.offset}: {message}")

    def whitespace(self):
        while self.offset < len(self.payload) and self.payload[self.offset] in b" \t\r\n":
            self.offset += 1

    def consume(self, expected: int):
        self.whitespace()
        if self.offset >= len(self.payload) or self.payload[self.offset] != expected:
            self.fail("unexpected delimiter")
        self.offset += 1

    def _utf8_end(self) -> int:
        """Validate a raw UTF-8 sequence without converting its code point."""
        start = self.offset
        first = self.payload[start]
        if 0xC2 <= first <= 0xDF:
            size, lower, upper = 2, 0x80, 0xBF
        elif 0xE0 <= first <= 0xEF:
            size = 3
            lower = 0xA0 if first == 0xE0 else 0x80
            upper = 0x9F if first == 0xED else 0xBF
        elif 0xF0 <= first <= 0xF4:
            size = 4
            lower = 0x90 if first == 0xF0 else 0x80
            upper = 0x8F if first == 0xF4 else 0xBF
        else:
            self.fail("invalid UTF-8")
        end = start + size
        if end > len(self.payload) or not lower <= self.payload[start + 1] <= upper:
            self.fail("invalid UTF-8")
        if any(not 0x80 <= byte <= 0xBF for byte in self.payload[start + 2:end]):
            self.fail("invalid UTF-8")
        return end

    def string(self) -> tuple[int, int]:
        self.whitespace()
        start = self.offset
        if start >= len(self.payload) or self.payload[start] != 34:
            self.fail("expected string")
        self.offset += 1
        while self.offset < len(self.payload):
            byte = self.payload[self.offset]
            if byte == 34:
                self.offset += 1
                return start, self.offset
            if byte == 92:
                self.offset += 1
                if self.offset >= len(self.payload):
                    self.fail("unterminated escape")
                escape = self.payload[self.offset]
                self.offset += 1
                if escape == ord("u"):
                    end = self.offset + 4
                    if end > len(self.payload) or any(
                        digit not in _HEX for digit in self.payload[self.offset:end]
                    ):
                        self.fail("invalid Unicode escape")
                    self.offset = end
                elif escape not in b'"\\/bfnrt':
                    self.fail("invalid string escape")
            elif byte < 0x20:
                self.fail("unescaped control character")
            elif byte >= 0x80:
                self.offset = self._utf8_end()
            else:
                self.offset += 1
        self.fail("unterminated string")

    def value(self, depth: int = 0) -> _Node:
        if depth > 100:
            self.fail("excessive nesting")
        self.whitespace()
        start = self.offset
        if start >= len(self.payload):
            self.fail("missing value")
        first = self.payload[start]
        if first == 123:
            self.offset += 1
            self.whitespace()
            fields = []
            keys = set()
            if self.offset < len(self.payload) and self.payload[self.offset] == 125:
                self.offset += 1
                return _Node(start, self.offset, "object")
            while True:
                key_start, key_end = self.string()
                # This is the only JSON decoding operation in the parser.
                key = json.loads(self.payload[key_start:key_end])
                if key in keys:
                    self.fail("duplicate object key")
                keys.add(key)
                self.consume(58)
                child = self.value(depth + 1)
                fields.append((key, key_start, key_end, child))
                self.whitespace()
                if self.offset < len(self.payload) and self.payload[self.offset] == 125:
                    self.offset += 1
                    return _Node(start, self.offset, "object", tuple(fields))
                self.consume(44)
        if first == 91:
            self.offset += 1
            self.whitespace()
            items = []
            if self.offset < len(self.payload) and self.payload[self.offset] == 93:
                self.offset += 1
                return _Node(start, self.offset, "array")
            while True:
                items.append(self.value(depth + 1))
                self.whitespace()
                if self.offset < len(self.payload) and self.payload[self.offset] == 93:
                    self.offset += 1
                    return _Node(start, self.offset, "array", items=tuple(items))
                self.consume(44)
        if first == 34:
            self.string()
            return _Node(start, self.offset, "scalar")
        number = _NUMBER.match(self.payload, self.offset)
        if number:
            self.offset = number.end()
            return _Node(start, self.offset, "scalar")
        for literal in (b"true", b"false", b"null"):
            if self.payload.startswith(literal, self.offset):
                self.offset += len(literal)
                return _Node(start, self.offset, "scalar")
        self.fail("invalid scalar token")

    def parse(self) -> _Node:
        root = self.value()
        self.whitespace()
        if self.offset != len(self.payload):
            self.fail("trailing bytes")
        if root.kind != "object":
            self.fail("report root must be an object")
        return root


def _matching_rule(path: tuple[str, ...], policies: tuple) -> tuple | None:
    for rule in policies:
        if len(path) == len(rule) and all(
            expected == "*" or actual == expected or
            (expected == "$series" and actual == path[0])
            for expected, actual in zip(rule, path)
        ):
            return rule
    return None


def _fields(node: _Node) -> dict:
    return {key: child for key, _, _, child in node.fields}


def _metadata_valid(report_name: str, payload: bytes, metadata: _Node) -> bool:
    if metadata.kind != "object":
        return False
    fields = _fields(metadata)
    expected = {"policy_version", "report", "original_sha256", "original_bytes",
                "redacted_path_counts", "replacement_meaning"}
    if set(fields) != expected:
        return False
    raw = lambda node: payload[node.start:node.end]
    for key, value in (("policy_version", POLICY_VERSION), ("report", report_name),
                       ("replacement_meaning", REPLACEMENT_MEANING)):
        if raw(fields[key]) != json.dumps(value).encode():
            return False
    if not re.fullmatch(rb'"[0-9a-f]{64}"', raw(fields["original_sha256"])):
        return False
    if not re.fullmatch(rb"(?:0|[1-9][0-9]*)", raw(fields["original_bytes"])):
        return False
    counts = fields["redacted_path_counts"]
    if counts.kind != "object":
        return False
    allowed = {".".join(rule) for rule in _POLICIES[report_name]}
    return all(key in allowed and re.fullmatch(rb"(?:0|[1-9][0-9]*)", raw(value))
               for key, _, _, value in counts.fields)


def sanitize_report(report_name: str, payload: bytes) -> bytes:
    """Return an explicitly marked public copy of one supported audit report.

    All retained scalar tokens are copied byte-for-byte. Source quotations,
    including numeric strings and already missing values, are replaced with
    ``null`` without conversion. Unknown report names and invalid JSON fail
    before producing output. An already sanitized report is returned unchanged
    only when its metadata and all applicable replacements are valid.
    """
    if report_name not in _POLICIES:
        raise ValueError("Unsupported public-report sanitization policy")
    if not isinstance(payload, bytes):
        raise TypeError("Report payload must be bytes")
    root = _Parser(payload).parse()
    policies = _POLICIES[report_name]
    fields = _fields(root)
    existing = fields.get(_METADATA_KEY)
    if existing is not None and not _metadata_valid(report_name, payload, existing):
        raise ValueError("Unrecognized public sanitization metadata")
    counts = Counter()

    def render(node: _Node, path: tuple[str, ...] = (), depth: int = 0) -> bytes:
        rule = _matching_rule(path, policies)
        if rule is not None:
            if existing is not None and payload[node.start:node.end] != b"null":
                raise ValueError("Sanitized report contains a source quotation")
            counts[".".join(rule)] += 1
            return b"null"
        if node.kind == "object":
            if not node.fields:
                return b"{}"
            return b"{\n" + b",\n".join(
                b"  " * (depth + 1) + payload[start:end] + b": " + render(child, path + (key,), depth + 1)
                for key, start, end, child in node.fields
            ) + b"\n" + b"  " * depth + b"}"
        if node.kind == "array":
            if not node.items:
                return b"[]"
            return b"[\n" + b",\n".join(
                b"  " * (depth + 1) + render(child, path + (str(index),), depth + 1)
                for index, child in enumerate(node.items)
            ) + b"\n" + b"  " * depth + b"]"
        return payload[node.start:node.end]

    rendered = render(root)
    if existing is not None:
        stored_counts = _fields(_fields(existing)["redacted_path_counts"])
        if set(stored_counts) != set(counts) or any(
            payload[node.start:node.end] != str(counts[key]).encode()
            for key, node in stored_counts.items()
        ):
            raise ValueError("Public sanitization path counts do not match the report")
        return payload
    metadata = {
        "policy_version": POLICY_VERSION,
        "report": report_name,
        "original_sha256": hashlib.sha256(payload).hexdigest(),
        "original_bytes": len(payload),
        "redacted_path_counts": dict(sorted(counts.items())),
        "replacement_meaning": REPLACEMENT_MEANING,
    }
    suffix = json.dumps({_METADATA_KEY: metadata}, indent=2, sort_keys=True,
                        allow_nan=False).encode()
    delimiter = b",\n" if root.fields else b"\n"
    return rendered[:-1].rstrip(b" \t\r\n") + delimiter + suffix[1:].lstrip(b"\n") + b"\n"
