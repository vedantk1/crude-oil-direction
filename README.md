# Crude oil direction research

A free, local WTI direction forecasting project with an audited historical dataset and an initial WTI-only model comparison over **five fixed US business days**. The 2023-onward final holdout remains closed; adding Brent is a future experiment.

Read the [data audit](docs/data-audit.md) for the findings, evidence limits, and recommended historical availability rules. The audit supports a qualified replay of ALFRED's recorded history from 2011-04-06 onward. Current EIA downloads alone do not preserve that history.

The [dataset specification](docs/dataset.md) describes the implemented cutoff, features, target, and label availability. The saved build contains **801 forecast rows**, from 2011-04-07 through 2026-10-08, with 789 first-available evaluation outcomes and 777 labels still available at the replay endpoint. These are dataset counts, not model results.

## Setup

Python 3.10 or later is required. These examples use Python 3.13. Free dependencies are pinned in `requirements.txt`: `xlrd` reads EIA's XLS files, while NumPy, SciPy, and scikit-learn support local CPU model evaluation.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run commands from the workspace root. Source files are stored locally under `data/`, which is excluded from Git. No credentials or paid services are required. The code does not apply an open-source license to third-party source data.

## Build and inspect the dataset

The dataset builder itself uses only Python's standard library; `xlrd` is needed for the separate EIA XLS audit. It runs offline against the saved ALFRED evidence:

```sh
.venv/bin/python -m oil_audit.dataset build
```

Each build checks the source hashes and writes `features.jsonl`, `label_events.jsonl`, `evaluation_labels.jsonl`, and `manifest.json` to `data/datasets/<content-id>/`. Identical code, configuration, and inputs reuse identical artifacts; changed inputs or code produce a new directory. The [dataset summary](reports/dataset_summary.json) points to the current build and contains coverage and missingness counts. Earlier datasets and source captures are preserved.

To export examples whose labels were available at a historical training cutoff:

```sh
.venv/bin/python -m oil_audit.dataset training-view \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --as-of 2020-04-23 --feature-set wti_brent \
  --output data/training-views/2020-04-23-wti-brent.jsonl
```

Use the directory printed by your build if its content ID differs. `--as-of` means 18:00 America/New_York under the same preceding-day information rule as forecasting. The export verifies dataset hashes, uses active historical labels, leaves optional missing inputs null, and exports only the declared feature columns. It fits no model. Choose a new output filename for another export; existing views are not overwritten.

Use `build --data-dir data/audit/vintages/<snapshot-directory>` for a newly downloaded snapshot. CLI paths must be workspace-relative. Do not train directly on the frozen evaluation-label file or on today's label state when simulating an older training cutoff.

## Run the development evaluation

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306
.venv/bin/python -m unittest discover -s tests -q
```

This runs on the CPU using the existing immutable dataset, with no new downloads or dataset rebuild. It starts training in 2011, refits before each 2017–2022 development forecast, and never predicts or scores the final holdout. Parameters and splits are fixed; there is no tuning or holdout CLI option.

On 301 scored development outcomes, accuracy was **49.5% for logistic regression, 49.5% for the training-majority baseline, and 50.2% for momentum**. Logistic regression had worse Brier and log-loss scores than the majority baseline. This experiment does not establish an improvement from the six WTI inputs.

See the [evaluation protocol and results](docs/evaluation.md) and [machine-readable report](reports/development_evaluation.json). Per-forecast predictions, fitted preprocessing diagnostics, and run hashes stay under ignored `data/evaluations/`. A repeated run with identical inputs reuses identical artifacts. All 99 tests pass.

## Reproduce the saved audit offline

These commands use the evidence retained from the 2026-10-13 audit. They check source hashes before reading values.

```sh
.venv/bin/python -m oil_audit inspect --manifest data/audit/current/20261013T055923606961Z/manifest.json
.venv/bin/python -m oil_audit.vintages
.venv/bin/python -m oil_audit.reconcile --manifest data/audit/current/20261013T055923606961Z/manifest.json
.venv/bin/python -m unittest discover -s tests -v
```

Results are saved to `reports/price_audit.json`, `reports/vintage_audit.json`, and `reports/source_reconciliation.json`. The report describes their interpretation. Running an offline audit does not change the original source files or their retrieval dates.

## Obtain new free source snapshots

```sh
.venv/bin/python -m oil_audit fetch
.venv/bin/python -m oil_audit fetch-releases
.venv/bin/python -m oil_audit.vintages --download
```

Each download creates a new timestamped directory and prints its location. Preserve earlier snapshots. Use the printed current manifest with `inspect --manifest`, the new vintage directory with `vintages --data-dir`, and all three locations with `reconcile --manifest ... --vintage-dir ... --release-dir ...`. Use `--output` to retain an additional result instead of replacing a report JSON file. Network access is needed only for downloads.

The ALFRED downloader submits the public download form and requests at most 450 listed vintages per batch. That website interface can change; unexpected responses fail rather than silently producing an empty dataset. The original audit captured four interval ZIPs and two independent snapshot ZIPs.

## Modules

- `sources`: public EIA downloads and snapshot provenance.
- `prices`: current XLS schema, coverage, gaps, and price checks.
- `vintages`: ALFRED downloads, interval integrity, revisions, and date-level reconstruction.
- `reconcile`: complete current-series comparison and four archived EIA Table 11 checks.
- `calendar`, `capture`: fixed US business days and conservative source-capture limits.
- `availability`: typed, validated archive intervals and quarantined records.
- `features`: separate benchmark lookups, dollar changes, coverage, age, and aligned spread.
- `labels`, `training`: versioned targets and cutoff-specific training selection.
- `dataset`: reproducible artifacts, manifests, summary, and CLI.

The separate `oil_eval` package contains the fixed protocol, holdout-filtered loader, models, metrics, chronological replay, and artifact runner. It consumes dataset artifacts without changing their source history.

The older `vintages.snapshot` helper remains an audit primitive: `strict=False` reproduces recorded anomalies; `strict=True` excludes observations on or after its query date and quarantines predated records. The dataset uses `availability` instead: an observation dated on its archive cutoff can be used at the following business day's issuance. Neither lookup establishes an exact historical intraday publication time.

## Next experiment and checkpoints

Completed checkpoints cover the source audit and dataset, the fixed chronological protocol and holdout safeguards, and the baseline-versus-WTI-logistic development comparison. The next planned experiment is to add Brent using the same development dates and keep the final holdout closed during that comparison. Final holdout evaluation belongs to a later, explicitly authorized step after model selection is finished.

Create a local commit after each coherent, verified milestone, with its relevant documentation updated. Keep source downloads and generated datasets under ignored `data/`. New branches and pushes require an explicit request.

The [chronological evaluation protocol](docs/evaluation.md) fixes initial training through 2016, development from 2017–2022, and a closed final holdout from 2023 onward. All training and preprocessing must respect each historical cutoff.
