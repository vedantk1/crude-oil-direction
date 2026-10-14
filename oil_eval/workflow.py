"""Build a new development reference chain from explicitly supplied source captures.

Run from the repository root. Existing reports, datasets and source captures are
read-only inputs; every output belongs to a fresh workspace-relative directory.
Real ALFRED training-use permission is unresolved. These real-data entry points
are historical technical recipes, not a permission grant. The public quickstart
is ``python -m oil_eval.demo`` with invented sources.
"""

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

from oil_audit.dataset import build, json_bytes, sha256, workspace_path
from oil_audit.inventory import run as audit_inventory
from oil_audit.inventory_sources import fetch as capture_inventories
from .benchmarks import run as run_benchmarks
from .runner import run as evaluate


IMMUTABLE_ROOTS = (Path("data/datasets"), Path("data/evaluations"),
                   Path("data/audits/inventories"), Path("reports"))


def checked_path(value: Path) -> Path:
    """Reject traversal and symlinks escaping the shared workspace."""
    path = workspace_path(str(value))
    if not path.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Workflow paths must remain inside the workspace")
    return path


def fresh_directory(value: Path) -> Path:
    path = checked_path(value)
    if path == Path(".") or path.exists():
        raise ValueError("Workflow output directory must be new: " + str(path))
    if any(path.resolve().is_relative_to(root.resolve()) for root in IMMUTABLE_ROOTS):
        raise ValueError("Workflow outputs cannot be placed inside historical artifact or report directories")
    if any((parent / "manifest.json").exists() for parent in path.parents if parent != Path(".")):
        raise ValueError("Workflow outputs cannot be placed inside an existing immutable artifact")
    return path


def _file_identity(path):
    payload = path.read_bytes()
    return {"path": str(path), "sha256": sha256(payload), "bytes": len(payload)}


def _save(path, manifest):
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        stream.write(json_bytes(manifest))
    temporary.replace(path)


def run(vintage_dir: Path, inventory_snapshot: Path, run_root: Path, *,
        report_dir: Path | None = None, evidence_manifest: Path | None = None,
        probe_manifest: Path | None = None, quarantine_manifest: Path | None = None,
        brent: bool = False, benchmarks: bool = False, window: bool = False,
        tree: bool = False, synthetic: bool | None = None) -> dict:
    """Discover newly built IDs and explicitly pass each authenticated reference.

    A failed stage leaves its completed predecessors and a failure checkpoint.
    Restart into a different fresh directory; no existing artifact is replaced.
    """
    if benchmarks and not brent:
        raise ValueError("Constant benchmark report requires the paired Brent comparison")
    vintage_dir, inventory_snapshot = map(checked_path, (vintage_dir, inventory_snapshot))
    inputs = {"vintage_manifest": vintage_dir / "manifest.json",
              "inventory_manifest": inventory_snapshot / "manifest.json"}
    optional = {}
    for name, value in (("evidence_manifest", evidence_manifest), ("probe_manifest", probe_manifest),
                        ("quarantine_manifest", quarantine_manifest)):
        optional[name] = None if value is None else checked_path(value)
        if optional[name] is not None:
            inputs[name] = optional[name]
    source_identity = {name: _file_identity(path) for name, path in inputs.items()}
    vintage_manifest = json.loads(inputs["vintage_manifest"].read_bytes())
    inventory_manifest = json.loads(inputs["inventory_manifest"].read_bytes())
    if not isinstance(vintage_manifest, list) or not vintage_manifest or not isinstance(inventory_manifest, dict):
        raise ValueError("Expected populated source capture manifests")
    for record in vintage_manifest:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Malformed ALFRED source capture record")
        source_file = checked_path(Path(record["path"]))
        if not source_file.resolve().is_relative_to(vintage_dir.resolve()):
            raise ValueError("ALFRED source record points outside its supplied capture directory")
    flags = {record.get("synthetic") is True for record in vintage_manifest}
    if len(flags) != 1 or (True in flags) != (inventory_manifest.get("synthetic") is True):
        raise ValueError("Source captures must consistently identify synthetic or research data")
    detected_synthetic = True in flags
    if synthetic is not None and synthetic != detected_synthetic:
        raise ValueError("Workflow kind differs from authenticated source provenance")
    synthetic = detected_synthetic
    root = fresh_directory(run_root)
    reports = fresh_directory(report_dir if report_dir is not None else root / "reports")
    artifact_roots = {name: root / name for name in ("datasets", "evaluations", "inventory_audits")}
    if any(reports.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(reports.resolve())
           for path in artifact_roots.values()):
        raise ValueError("Workflow report directory must be separate from immutable artifact roots")
    if reports == root or root.resolve().is_relative_to(reports.resolve()):
        raise ValueError("Workflow report directory cannot contain the run root")
    if any(path.resolve().is_relative_to(root.resolve()) or path.resolve().is_relative_to(reports.resolve())
           for path in inputs.values()):
        raise ValueError("Workflow outputs must be separate from source captures")
    root.mkdir(parents=True)
    reports.mkdir(parents=True)
    state = {
        "schema_version": 1, "kind": "synthetic_demo" if synthetic else "new_capture_research_replay",
        "interpretation": ("Invented data for software verification only; scores have no market meaning."
                           if synthetic else "New capture replay; not an exact reconstruction of earlier project runs."),
        "status": "running", "holdout_evaluated": False,
        "run_root": str(root), "report_directory": str(reports),
        "reports": {}, "development_as_of": "2022-12-31",
        "inputs": source_identity, "stages": {},
        "requested_comparisons": {"brent": brent, "benchmarks": benchmarks, "window": window, "tree": tree},
    }
    manifest_path = root / "workflow.json"
    _save(manifest_path, state)

    def stage(name, action):
        state["active_stage"] = name
        _save(manifest_path, state)
        result = action()
        report = reports / (name + ".json")
        entry = {"report": _file_identity(report)}
        for key in ("dataset_directory", "evaluation_directory", "audit_directory", "benchmark_directory",
                    "run_id", "audit_id", "dataset_id"):
            if key in result:
                entry[key] = result[key]
        if name == "dataset":
            built_manifest = json.loads((Path(result["dataset_directory"]) / "manifest.json").read_bytes())
            entry["dataset_id"] = built_manifest["dataset_id"]
        state["stages"][name] = entry
        state["reports"][name] = str(report)
        if name == "dataset":
            state["dataset_directory"] = result["dataset_directory"]
        _save(manifest_path, state)
        return result

    try:
        dataset = stage("dataset", lambda: build(vintage_dir, artifact_roots["datasets"],
                                                  report=reports / "dataset.json",
                                                  maximum_as_of=date(2022, 12, 31)))
        dataset_dir = Path(dataset["dataset_directory"])
        wti_report = reports / "wti.json"
        stage("wti", lambda: evaluate(dataset_dir, artifact_roots["evaluations"], wti_report))
        audit_report = reports / "inventory_audit.json"
        stage("inventory_audit", lambda: audit_inventory(
            inventory_snapshot, artifact_roots["inventory_audits"], audit_report,
            reference_report=wti_report, **optional))
        inventory_report = reports / "inventory_comparison.json"
        regularization_report = reports / "inventory_regularization.json"
        options = {"reference_report": wti_report, "inventory_audit_report": audit_report,
                   "inventory_reference_report": inventory_report,
                   "regularization_reference_report": regularization_report}
        for name, experiment in (
            ("inventory_comparison", "inventory-comparison"),
            ("inventory_regularization", "inventory-regularization"),
        ):
            stage(name, lambda name=name, experiment=experiment: evaluate(
                dataset_dir, artifact_roots["evaluations"], reports / (name + ".json"),
                experiment=experiment, **options))
        if brent:
            stage("brent_comparison", lambda: evaluate(
                dataset_dir, artifact_roots["evaluations"], reports / "brent_comparison.json",
                experiment="brent-comparison", reference_report=wti_report))
        if benchmarks:
            stage("benchmarks", lambda: run_benchmarks(
                wti_report, reports / "brent_comparison.json", artifact_roots["evaluations"],
                reports / "benchmarks.json"))
        for enabled, name, experiment in (
            (window, "inventory_window_comparison", "inventory-window-comparison"),
            (tree, "inventory_tree_comparison", "inventory-tree-comparison"),
        ):
            if enabled:
                stage(name, lambda name=name, experiment=experiment: evaluate(
                    dataset_dir, artifact_roots["evaluations"], reports / (name + ".json"),
                    experiment=experiment, **options))
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        _save(manifest_path, state)
        raise
    state.pop("active_stage", None)
    state["status"] = "complete"
    _save(manifest_path, state)
    return state


def capture_and_run(run_root: Path, *, max_workers: int = 12, **options) -> dict:
    """Explicit opt-in free downloads into a new root, then a bounded replay.

    ZIP bytes may contain later archive history. The bounded dataset builder
    gates observation and vintage dates before decoding any later value.
    """
    if type(max_workers) is not int or not 1 <= max_workers <= 12:
        raise ValueError("Inventory worker count must be between 1 and 12")
    root = fresh_directory(run_root)
    root.mkdir(parents=True)
    sources = root / "sources"
    sources.mkdir()
    state = {"schema_version": 1, "kind": "new_free_capture", "status": "capturing_vintages",
             "holdout_evaluated": False, "development_as_of": "2022-12-31"}
    checkpoint = root / "capture_workflow.json"
    _save(checkpoint, state)
    try:
        vintage_root = sources / "vintages"
        subprocess.run([sys.executable, "-m", "oil_audit.vintages", "--download", "--capture-only",
                        "--data-dir", str(vintage_root), "--output", str(sources / "vintage_capture_audit.json")], check=True)
        manifests = sorted(vintage_root.glob("*/manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Expected exactly one newly captured ALFRED source manifest")
        vintage_dir = manifests[0].parent
        state.update(status="capturing_inventories", vintage_manifest=_file_identity(manifests[0]))
        _save(checkpoint, state)
        inventory_manifest = capture_inventories(sources / "inventories", max_workers=max_workers)
        state.update(status="replaying", inventory_manifest=_file_identity(inventory_manifest))
        _save(checkpoint, state)
        result = run(vintage_dir, inventory_manifest.parent, root / "replay", **options)
    except Exception as exc:
        state.update(failed_stage=state["status"], status="failed", error_type=type(exc).__name__)
        _save(checkpoint, state)
        raise
    state.update(status="complete", workflow_manifest=str(root / "replay/workflow.json"))
    _save(checkpoint, state)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage-dir", type=workspace_path)
    parser.add_argument("--inventory-snapshot", type=workspace_path)
    parser.add_argument("--download", action="store_true",
                        help="historical ALFRED capture/replay route; training-use permission must be resolved first")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--run-root", type=workspace_path, required=True)
    parser.add_argument("--report-dir", type=workspace_path)
    parser.add_argument("--evidence-manifest", type=workspace_path)
    parser.add_argument("--probe-manifest", type=workspace_path)
    parser.add_argument("--quarantine-manifest", type=workspace_path)
    for name in ("brent", "benchmarks", "window", "tree"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    options = vars(args).copy()
    download, workers = options.pop("download"), options.pop("max_workers")
    if download:
        if args.vintage_dir is not None or args.inventory_snapshot is not None:
            parser.error("--download creates its own sources; omit --vintage-dir and --inventory-snapshot")
        options.pop("vintage_dir")
        options.pop("inventory_snapshot")
        result = capture_and_run(max_workers=workers, **options)
    else:
        if args.vintage_dir is None or args.inventory_snapshot is None:
            parser.error("offline replay requires --vintage-dir and --inventory-snapshot")
        result = run(**options)
    print(json.dumps({"status": result["status"], "kind": result["kind"],
                      "workflow_manifest": str(Path(result["run_root"]) / "workflow.json"),
                      "holdout_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
