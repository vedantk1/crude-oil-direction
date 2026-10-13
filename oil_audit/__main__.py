"""Run from the workspace root: python -m oil_audit --help."""

import argparse
import json
from pathlib import Path

from .prices import run_audit
from .sources import RELEASE_SOURCES, fetch_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    fetch = subcommands.add_parser("fetch", help="Download new public source snapshot")
    fetch.add_argument("--output", type=Path, default=Path("data/audit/current"))
    releases = subcommands.add_parser("fetch-releases", help="Download the four sampled historical price tables")
    releases.add_argument("--output", type=Path, default=Path("data/audit/releases"))
    audit = subcommands.add_parser("inspect", help="Audit saved source bytes offline")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=Path("reports/price_audit.json"))
    args = parser.parse_args()
    if args.output.is_absolute() or (args.command == "inspect" and args.manifest.is_absolute()):
        parser.error("Use workspace-relative paths")
    if args.command == "fetch":
        print(fetch_snapshot(args.output).as_posix())
    elif args.command == "fetch-releases":
        print(fetch_snapshot(args.output, RELEASE_SOURCES).as_posix())
    else:
        result = run_audit(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(args.output.as_posix())


if __name__ == "__main__":
    main()
