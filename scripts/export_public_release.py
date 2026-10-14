"""Create an optional source ZIP from tracked HEAD files."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oil_audit.public_reports import sanitize_report


ARCHIVE_PREFIX = "crude-oil-direction"
ARCHIVE_NAME = ARCHIVE_PREFIX + ".zip"
MANIFEST_NAME = "PUBLIC_RELEASE_MANIFEST.json"
PUBLIC_REPORTS = {
    "reports/price_audit.json": "price_audit.json",
    "reports/vintage_audit.json": "vintage_audit.json",
    "reports/source_reconciliation.json": "source_reconciliation.json",
}
PRIVATE_COMPONENTS = frozenset({
    ".git", ".venv", ".audit-deps", ".agents", ".codex", "__pycache__",
    "data", "private", "backups", "node_modules", "venv",
})
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(
    prefix + str(index) for prefix in ("COM", "LPT") for index in range(1, 10))})


def _git(*arguments, input_bytes=None):
    result = subprocess.run(["git", *arguments], input=input_bytes,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git could not read the current repository snapshot")
    return result.stdout


def _source_path(value):
    """Reject portable-path hazards and private files rather than omitting them."""
    if (not value or value.startswith("/") or any(character in value for character in '\\:<>|?*')
            or any(unicodedata.category(character) in ("Cc", "Cf") for character in value)):
        raise ValueError("Unsafe tracked filename in public release")
    parts = value.split("/")
    if any(not part or part in (".", "..") for part in parts):
        raise ValueError("Unsafe tracked filename in public release")
    if any(part.endswith((".", " ")) or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED
           for part in parts):
        raise ValueError("Unsafe tracked filename in public release")
    if any(part.lower() in PRIVATE_COMPONENTS or part.lower() == ".env"
           or part.lower().startswith(".env.") for part in parts):
        raise ValueError("Tracked private or data paths cannot enter a public release")
    if unicodedata.normalize("NFC", value).casefold() == MANIFEST_NAME.casefold():
        raise ValueError("Tracked filename collides with the generated release manifest")
    return value


def _output_path(value):
    path = Path(value)
    if (path.is_absolute() or ".." in path.parts or len(path.parts) < 3
            or path.parts[:2] != ("data", "releases")
            or not path.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError("Use a fresh workspace-relative directory under data/releases")
    if path.exists() or path.is_symlink():
        raise ValueError("Existing release outputs are preserved; choose a fresh directory")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Release output paths must not traverse symlinks")
    return path


def _tracked_head():
    if _git("rev-parse", "--show-prefix").strip():
        raise ValueError("Run the public release export from the repository root")
    if _git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Public export requires a clean working tree and index; commit changes first")
    commit = _git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    if not _OBJECT_ID.fullmatch(commit):
        raise ValueError("Invalid source commit identity")
    entries = []
    for entry in _git("ls-tree", "-r", "-z", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        try:
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = _source_path(raw_path.decode("utf-8"))
        except (UnicodeError, TypeError) as error:
            raise ValueError("Unsafe tracked filename in public release") from error
        if mode not in ("100644", "100755") or object_type != "blob":
            raise ValueError("Public release accepts regular tracked files only; no symlinks or submodules")
        if not _OBJECT_ID.fullmatch(object_id):
            raise ValueError("Invalid tracked blob identity")
        entries.append((path, object_id))
    if len({unicodedata.normalize("NFC", path).casefold() for path, _ in entries}) != len(entries):
        raise ValueError("Duplicate tracked filenames in public release")
    normalized = {unicodedata.normalize("NFC", path).casefold() for path, _ in entries}
    if any("/".join(path.split("/")[:index]) in normalized
           for path in normalized for index in range(1, len(path.split("/")))):
        raise ValueError("Tracked file and directory names collide in portable public release")
    if not set(PUBLIC_REPORTS).issubset(path for path, _ in entries):
        raise ValueError("Public release requires all three sanitized price audit reports")
    return commit, sorted(entries)


def _blob_payloads(entries):
    response = _git("cat-file", "--batch", input_bytes="".join(
        object_id + "\n" for _, object_id in entries).encode("ascii"))
    cursor, payloads = 0, {}
    for path, object_id in entries:
        end = response.find(b"\n", cursor)
        if end < 0:
            raise ValueError("Incomplete tracked blob response")
        header = response[cursor:end].split(b" ")
        if (len(header) != 3 or header[:2] != [object_id.encode("ascii"), b"blob"]
                or not header[2].isdigit()):
            raise ValueError("Invalid tracked blob response")
        size = int(header[2])
        cursor = end + 1
        payload = response[cursor:cursor + size]
        if len(payload) != size or response[cursor + size:cursor + size + 1] != b"\n":
            raise ValueError("Incomplete tracked blob response")
        payloads[path] = payload
        cursor += size + 1
    if cursor != len(response):
        raise ValueError("Unexpected bytes after tracked blob response")
    return payloads


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _archive(payloads, manifest):
    buffer = io.BytesIO()
    files = {**payloads, MANIFEST_NAME: manifest}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, payload in sorted(files.items()):
            entry = zipfile.ZipInfo(ARCHIVE_PREFIX + "/" + path, (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.create_version = 20
            entry.extract_version = 20
            entry.external_attr = 0o100644 << 16
            entry.internal_attr = 0
            entry.extra = b""
            entry.comment = b""
            archive.writestr(entry, payload)
    return buffer.getvalue()


def export_public_release(output_root):
    """Validate everything before creating a deterministic, history-free export.

    The manifest identifies the source commit, without its author,
    committer, timestamps or messages. It is provenance, not Git history.
    """
    output_root = _output_path(output_root)
    commit, entries = _tracked_head()
    payloads = _blob_payloads(entries)
    for path, report_name in PUBLIC_REPORTS.items():
        payload = payloads[path]
        if sanitize_report(report_name, payload) != payload:
            raise ValueError("Public audit reports must already carry valid public_sanitization metadata")
    manifest = {
        "schema_version": 1,
        "artifact_type": "public-source-release",
        "source_commit": commit,
        "archive_prefix": ARCHIVE_PREFIX,
        "archive_name": ARCHIVE_NAME,
        "file_count": len(payloads),
        "files": {path: {"bytes": len(payload), "sha256": _sha256(payload)}
                  for path, payload in sorted(payloads.items())},
        "policy": "clean tracked HEAD regular files only; no Git history, ignored files, private paths or symlinks; sanitized public audit reports; fixed ZIP timestamps and permissions",
    }
    manifest_payload = _json_bytes(manifest)
    archive_payload = _archive(payloads, manifest_payload)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    with (output_root / "manifest.json").open("xb") as destination:
        destination.write(manifest_payload)
    with (output_root / ARCHIVE_NAME).open("xb") as destination:
        destination.write(archive_payload)
    return {
        "output_directory": output_root.as_posix(),
        "archive": (output_root / ARCHIVE_NAME).as_posix(),
        "source_commit": commit,
        "file_count": len(payloads),
        "archive_bytes": len(archive_payload),
        "archive_sha256": _sha256(archive_payload),
        "manifest_sha256": _sha256(manifest_payload),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="fresh workspace-relative directory under data/releases")
    args = parser.parse_args()
    print(json.dumps(export_public_release(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
