"""Preserve private audit bytes and write explicit public summaries."""

import hashlib
from pathlib import Path

from .public_reports import sanitize_report


def _workspace_path(path):
    path = Path(path)
    if (path.is_absolute() or ".." in path.parts or path == Path(".")
            or not path.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError("Audit output must stay workspace-relative")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Audit output and private archives cannot use symlink paths")
    return path


def preserve_report(report_name, payload, archive_root=Path("data/private-report-archive")):
    """Store the original unchanged under its checksum, without reading values."""
    archive_root = _workspace_path(archive_root)
    if not archive_root.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Private originals must stay in ignored data/")
    checksum = hashlib.sha256(payload).hexdigest()
    destination = archive_root / checksum / report_name
    _workspace_path(destination)
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError("Private original checksum archive differs; refusing replacement")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as stream:
            stream.write(payload)
    return destination


def write_audit_report(report_name, payload, output):
    """Public report paths receive redactions; raw output is restricted to data/."""
    output = _workspace_path(output)
    public = output.resolve().is_relative_to(Path("reports").resolve())
    private = output.resolve().is_relative_to(Path("data").resolve())
    if not public and not private:
        raise ValueError("Use reports/ for a public summary or ignored data/ for private raw output")
    # Validate the full structure and policy before creating any file.
    sanitized = sanitize_report(report_name, payload)
    if private:
        if output.exists() and output.read_bytes() != payload:
            raise ValueError("Choose a new private report path; existing originals are preserved")
        selected = payload
    else:
        if output.exists():
            previous = output.read_bytes()
            if sanitize_report(report_name, previous) != previous:
                preserve_report(report_name, previous)
        preserve_report(report_name, payload)
        selected = sanitized
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(selected)
    temporary.replace(output)
    return output
