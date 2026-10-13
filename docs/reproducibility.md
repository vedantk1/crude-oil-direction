# Reproduction and fresh checkouts

Three workflows answer different questions. None needs paid data, an API key,
a GPU or an earlier project's local run IDs. Run from the repository root.

## Install and demonstrate

Python 3.13 is the tested interpreter. Dependencies are pinned in
`requirements.txt`; bootstrap creates a new environment and imports the numeric
runtime to check installation. It refuses to replace an existing environment.

```sh
python3.13 scripts/bootstrap.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m oil_eval.demo --run-root data/demo
```

On Windows use `python` with Python 3.13 and `.venv\Scripts\python.exe`.
On macOS arm64 bootstrap selects the official same-version SciPy 1.15.3
`macosx_12_0_arm64` wheel and checks SHA256
`7e11270a000969409d37ed399585ee530b9ef6aa99d50c019de4cb01e8e54e62`.
This avoids the loader problem observed with the macOS 14 build; existing local
environments and the earlier isolated workaround remain unchanged. Other
platforms use their compatible PyPI builds. No source compilation is required
on the tested platform. Native build fingerprints distinguish numerical runtimes.

The demonstration creates small invented ALFRED/EIA-style captures, builds a
bounded dataset, creates new WTI and inventory references, evaluates C=1 and
C=0.1 inventory models, and compares Brent and constants. `--tree` adds the fixed
tree comparison. It deliberately contains invalid future-value sentinels to
check admission boundaries. Demonstration metrics have **no market meaning**.
CI runs the tests and demo on Linux/Python 3.13; it makes no market-data requests.

## New free captures

```sh
.venv/bin/python -m oil_eval.workflow --download --run-root data/new-capture
```

This explicitly downloads free, key-free ALFRED archives and EIA inventory
releases for 2011–2022, then builds and evaluates a new reference chain.
Network access and functioning public sources are required. Inventory capture
can require hundreds of requests; worker count is bounded by `--max-workers`.
Sources go under `data/new-capture/sources/` and replay outputs under
`data/new-capture/replay/`. Capture and replay checkpoints remain if a stage
fails; inspect them and restart with a new root. Existing roots are refused.

ALFRED archives may contain later history. Capture-only mode saves and verifies
raw bytes without running the numerical audit. The dataset builder then gates
observation and vintage dates **before decoding values**, bounds the supported
replay at 2022-12-31 and masks interval ends reaching later history. Forecasts,
training labels and scoring labels after this boundary cannot enter evaluation.
The final holdout is neither fitted, predicted nor scored.

Add `--brent --benchmarks --window --tree` to reproduce the fixed supplementary
development comparisons. These settings repeat predefined experiments; they
do not search parameters. A new capture can contain changed archive history,
metadata or availability evidence. Its identities and results need not match
retained project reports, even if historical values look unchanged.

## Offline replay from retained evidence

Copy an authorized source snapshot into the checkout, preserving its relative
paths and manifest checksums. Supply its directories explicitly:

```sh
.venv/bin/python -m oil_eval.workflow \
  --vintage-dir data/audit/vintages \
  --inventory-snapshot data/audit/inventories/RETAINED_CAPTURE \
  --run-root data/offline-replay --brent --benchmarks --window --tree
```

`RETAINED_CAPTURE` denotes the directory holding that snapshot's manifest;
replace it with your actual relative path. Optional `--evidence-manifest`,
`--probe-manifest` and `--quarantine-manifest` attach previously captured EIA
cross-checks. Without them, absence is recorded and evidence claims are narrower.
No historical report ID or prior evaluation directory is needed. The workflow
discovers each built ID, authenticates the WTI reference, creates an audit tied
to that reference, then passes new C=1/C=0.1 references explicitly.

`workflow.json` records input hashes, dataset directory, stage IDs, report paths,
status and the closed-holdout flag. Reports are separate from immutable
content-addressed artifacts. Existing sources, reports and runs are preserved.
The code/source/config/runtime identities make any changes visible.

An offline rerun of retained sources checks historical values and predictions.
It is **not byte-for-byte recreation of an old run** after code, path metadata,
native builds or configuration have changed. Exact historical artifacts require
the original commit, source bytes, configuration and numeric build. Saved
historical reports alone are insufficient to reconstruct omitted source files.

## Predefined availability-delay check

For a built workflow use the dataset directory and reports listed in its
`workflow.json`; pass all four reference reports to the dedicated command:

```sh
.venv/bin/python -m oil_eval.delay_runner \
  --dataset-dir data/offline-replay/datasets/DATASET_ID \
  --vintage-dir data/audit/vintages \
  --output-root data/availability-delay/evaluations \
  --report data/availability-delay/report.json \
  --reference-report data/offline-replay/reports/wti.json \
  --inventory-audit-report data/offline-replay/reports/inventory_audit.json \
  --inventory-reference-report data/offline-replay/reports/inventory_comparison.json \
  --regularization-reference-report data/offline-replay/reports/inventory_regularization.json
```

`DATASET_ID` is the ID returned by your build. This is one fixed timing
sensitivity check, not another model search. It reconstructs original WTI
inputs exactly, reproduces expanding controls, applies one extra US business-day
delay to WTI, inventories and active training-label states, and retains original
forecast dates and scoring targets. See [evaluation](evaluation.md) for the
locked protocol and results.
