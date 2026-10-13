# Crude oil direction research

A free, local WTI direction forecasting project with an audited historical dataset and chronological model comparisons over **five fixed US business days**. WTI-only, WTI/Brent, and WTI/inventory logistic regressions have been evaluated on the same development dates. The 2023-onward final holdout remains closed.

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
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --report data/evaluations/wti_reproduction.json
.venv/bin/python -m unittest discover -s tests -q
```

This runs on the CPU using the existing immutable dataset, with no new downloads or dataset rebuild. It starts training in 2011, refits before each 2017–2022 development forecast, and never predicts or scores the final holdout. Parameters and splits are fixed; there is no tuning or holdout CLI option.

On the same 301 scored development outcomes:

| Model | Accuracy | Brier score | Log loss |
| --- | ---: | ---: | ---: |
| Always up | 55.81% | — | — |
| Always not up | 44.19% | — | — |
| Constant probability 0.5 | 55.81% | 0.2500 | 0.6931 |
| Training-majority baseline | 49.50% | 0.2512 | 0.6955 |
| Five-day momentum | 50.17% | — | — |
| WTI-only logistic | 49.50% | 0.2578 | 0.7179 |
| WTI + Brent logistic | 51.16% | 0.2666 | 0.7336 |
| WTI + inventory logistic | 52.49% | 0.2574 | 0.7174 |

Against WTI-only, Brent adds five correct predictions (**+1.66 percentage points**) while worsening probability scores. Inventories add nine (**+2.99 points**) with very small Brier/log-loss improvements. Inventory accuracy improves in four of six development years. All three fitted models trail always-up on accuracy and constant 0.5 on probability scores. The constant direction rules each have 50% balanced accuracy; always-up's 55.81% headline score reflects the development set's greater number of up outcomes.

See the [evaluation protocol and results](docs/evaluation.md), [original WTI report](reports/development_evaluation.json), [Brent comparison](reports/brent_comparison.json), [inventory comparison](reports/inventory_comparison.json), and [supplemental benchmarks](reports/development_benchmarks.json). Per-forecast predictions, fitted preprocessing diagnostics, and run hashes stay under ignored `data/evaluations/`. Repeated runs with identical inputs reuse identical artifacts. All **247 tests pass**, including inventory feature, preprocessing, cohort, and provenance checks. The WTI command above records a new reproduction report while preserving the original reference.

The Brent comparison authenticates the saved WTI run and requires its exact forecast dates and scoring labels before fitting:

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment brent-comparison
```

It uses the same eligible historical rows for both logistic regressions, with six WTI inputs versus fourteen WTI/Brent inputs. The default result is `reports/brent_comparison.json`; the original WTI report and artifacts are preserved. All 310 WTI training-input hashes and original model outputs reproduced exactly; Brent eligibility removed no historical training rows. The fixed protocol was committed as `77c24e9` before computing Brent results.

To reproduce the added benchmarks from authenticated saved development artifacts, without fitting models or opening the source dataset:

```sh
.venv/bin/python -m oil_eval.benchmarks
```

This creates a separate content-addressed analysis and `reports/development_benchmarks.json`, preserving both original reports. Always-up and always-not-up are direction rules with no assigned probability; the neutral 0.5 probability predicts up under the existing tie convention. These benchmarks were added retrospectively and are now also included in future development replay summaries.

## Audit free inventory data

The [2011–2022 EIA inventory audit](docs/inventory-audit.md) checked 597 intended issues for commercial crude and Cushing stocks. It admitted 596 CSV issues, with inventory history available for 309 of the original 310 development dates. Early 2011 coverage remains incomplete. A stale July 2019 CSV is quarantined; a correct lower-precision PDF is documented as a possible later fallback. The October 2016 definition change and delayed releases are handled explicitly. Archived values support a qualified replay; they do not prove immutable first releases.

The audit adds no model inputs and fits no model. Its [report](reports/inventory_audit.json) records source checks, overlap comparisons, gaps, and per-series coverage. The final holdout stays closed. Reproduce it offline with the exact evidence paths in the audit document, or capture a new bounded snapshot:

```sh
.venv/bin/python -m oil_audit.inventory_sources
```

Use the printed snapshot directory with `python -m oil_audit.inventory --snapshot-dir <directory>`. Optional evidence manifests add independent Table 1 comparisons, early-2011 failed probes, and quarantined bytes. Captures and audit artifacts remain under ignored `data/`; downloads need no credentials or payment.

## Run the inventory comparison

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment inventory-comparison
```

The fixed eleven-input model adds weekly and four-week changes for commercial crude and Cushing, plus shared report age, to the original six WTI columns. It uses the same training examples and development dates as WTI-only. Early missing history and quarantined stock changes remain explicit nulls, with preprocessing fitted inside each training window. The saved WTI outputs must reproduce exactly. See the [protocol](docs/evaluation.md#fixed-inventory-comparison) for cutoff, definition, provenance, and missing-data rules.

The protocol and implementation were committed as `1ac70a8` before computing results. The run retained all 310 forecast dates, 301 scored outcomes, and original training rows. All WTI hashes and outputs reproduced exactly. The 607 raw inventory feature rows preserve sixteen early rows with no retained history, missing lookbacks/definition-compatible references, and the quarantined forecast's four null changes with known age. It consumes the existing dataset and audit offline, writes a separate `reports/inventory_comparison.json`, and preserves the final holdout and historical reports.

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
- `inventory_sources`, `inventory_tables`, `inventory`: bounded EIA release captures, strict stock-table parsing, and qualified offline inventory auditing.

The separate `oil_eval` package contains the fixed protocol, holdout-filtered loader, models, metrics, reference validation, chronological replay, and artifact runner. It consumes dataset artifacts without changing their source history.

The older `vintages.snapshot` helper remains an audit primitive: `strict=False` reproduces recorded anomalies; `strict=True` excludes observations on or after its query date and quarantines predated records. The dataset uses `availability` instead: an observation dated on its archive cutoff can be used at the following business day's issuance. Neither lookup establishes an exact historical intraday publication time.

## Next experiment and checkpoints

Completed checkpoints cover the source audit and dataset, fixed chronological protocol and holdout safeguards, WTI/Brent comparison, constant benchmarks, inventory archive audit, and inventory comparison on identical development dates with explicit missingness. Other possible development checks include a longer publication delay and scoring-policy sensitivity. Final holdout evaluation belongs to a later, explicitly authorized step after model selection is finished.

Create a local commit after each coherent, verified milestone, with its relevant documentation updated. Keep source downloads and generated datasets under ignored `data/`. New branches and pushes require an explicit request.

The [chronological evaluation protocol](docs/evaluation.md) fixes initial training through 2016, development from 2017–2022, and a closed final holdout from 2023 onward. All training and preprocessing must respect each historical cutoff.
