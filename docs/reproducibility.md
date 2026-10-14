# Reproduction and fresh checkouts

The synthetic workflow is the current public quickstart. Historical real-data
recipes remain below as technical records, **not source-permission clearance**.
Do not run new ALFRED training while its training-use terms are unresolved.
See [data rights](../DATA_LICENSE.md) and the [alternative-source audit](free-training-data-audit.md).
No paid data, API key or GPU is needed for the synthetic demo. Run from the repository root.

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

## Historical real-data capture recipe — permission unresolved

```sh
.venv/bin/python -m oil_eval.workflow --download --run-root data/new-capture
```

This historically tested command downloads key-free ALFRED archives and EIA inventory
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

## Historical retained-evidence recipe — permission unresolved

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

## Dedicated availability-delay command — permission unresolved

The generic evaluator now rejects `inventory-availability-delay` before any
source read or write. Only this dedicated command implements the delay; using
it with original ALFRED data still requires resolved training-use permission.
For an appropriately authorized built workflow use the dataset directory and reports listed in its
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


## Recorded verification

Checkpoint `f4ca7fa` was extracted into a clean checkout with no saved datasets,
evaluation runs or environment. The documented bootstrap installed the pinned
runtime; all 349 tests at that checkpoint and the synthetic demo through tree
comparison passed. Only then were copies of retained source captures supplied.
The complete nine-stage replay built new IDs and references, retained all 310
forecasts/301 scored outcomes, and reproduced every direction prediction.
Regularization, rolling-window and tree prediction files were byte-identical to
retained results. Older WTI/Brent/C=1 references differ only by tiny native-build
floating-point amounts (maximum probability difference 1.784e-13), with no
changed directions. Those older references predate native fingerprint recording.

The final implementation passes 375 tests. The separately checkpointed delay
result was independently verified and reproduced byte-identically. All 727
pre-existing source/data/report files retain their checksums. CI configuration
has been added; no hosted GitHub run or push was performed. The
[verification report](../reports/reproducibility_check.json) records exact stage
identities, checks and the live source-download check separately.


The live `--download` workflow also completed with newly captured ALFRED archives
and all 597 intended EIA inventory issues: 596 tables are usable, with the known
stale July 2019 issue excluded. The pipeline created its own bounded dataset,
WTI reference, inventory audit and C=1/C=0.1 references. All 310 development
forecasts and 301 scored outcomes were retained. C=0.1 prediction bytes match
the retained run exactly, while capture/dataset/run identities differ as expected.
Every stage report and artifact checksum verifies. This demonstrates that the
free download path worked at this check; it does not guarantee future source
availability or erase the historical-publication limitations. The holdout stayed
closed throughout numerical preparation and evaluation.
