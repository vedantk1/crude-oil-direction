# Crude Oil Direction Prediction

A free, CPU-only forecasting project for predicting whether WTI spot rises over
**five fixed US business days**. It reconstructs source availability, trains on
historically available labels, and compares simple models chronologically.
The 2023-onward final holdout remains closed.

The project uses its original repository and full development history. Code,
documentation and historical aggregate results belong in Git; downloaded data,
private backups and full model artifacts stay under ignored `data/`. See the
[development history](docs/development-history.md) and
[publication record](docs/public-release.md).

**This research phase is frozen. No fitted model has demonstrated an advantage
over the simple probability benchmarks.** The project is useful as a reproducible
forecasting investigation, including its negative findings; it is not a validated
price forecast or trading system. See the [research card](docs/research-card.md).

## Start from a fresh checkout

Use Python 3.13 and run from the repository root:

```sh
python3.13 scripts/bootstrap.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m oil_eval.demo --run-root data/demo
```

On Windows use a Python 3.13 `python` and `.venv\Scripts\python.exe`.
Bootstrap installs pinned dependencies into a new environment and refuses to
modify an existing one. On macOS arm64 it verifies and selects an official
compatible SciPy wheel. The demo uses invented data, fits real models, and needs
no market downloads or API key. Its metrics have no market meaning.

The public quickstart uses synthetic data only. **New ALFRED-based training is
not cleared:** the current FRED legal page includes a machine-learning training
restriction and its applicability includes ALFRED. Free access and ignored raw
files do not establish training permission. See [source rights](DATA_LICENSE.md).

The [free-training-data audit](docs/free-training-data-audit.md) identifies
licensed World Bank monthly WTI/Brent data and EIA's own survey aggregates as
candidates for a separate phase. None is a verified replacement for this exact
daily WTI/vintage experiment. No replacement dataset or model has been trained.

[Reproducibility](docs/reproducibility.md) distinguishes the working synthetic
demo, completed historical source replays and permission-restricted real-data
commands. The existing model results are retained research findings; they do not
assert current source-use clearance.

## Retained development findings

All comparisons use the same 301 scored outcomes from 310 forecasts in
2017–2022, with nine explicitly unscored states. The original expanding models
use 295–596 historically available examples per fit. Later experiments are
retrospective development choices, not independent confirmatory tests.

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
| WTI + inventory logistic, C=0.1 | 53.16% | 0.2563 | 0.7126 |
| WTI + inventory, C=0.1, four-year rolling | 50.83% | 0.2697 | 0.7866 |
| WTI + inventory, small boosted trees | 50.17% | 0.2561 | 0.7057 |

Lower Brier score and log loss are better. Always-up accuracy reflects more up
outcomes; its balanced accuracy is 50%. Brent improved direction accuracy but
worsened probabilities. Inventories and stronger regularization made modest
improvements over WTI-only. Rolling training worsened both direction and
probabilities. Trees improved some probability scores but lost direction
accuracy and remained worse than simple probability benchmarks. These results
do not establish forecasting skill or profitability.

The predefined extra availability-delay check is recorded separately in
[evaluation](docs/evaluation.md). It tests timing sensitivity without choosing a
new model from the result. It scores 53.82% with slightly better probability scores, but changes 106 of
301 scored directions and still trails the simple probability benchmarks. The
original forecasts, targets and scoring labels stay fixed, while both inputs and
training labels receive the safety margin. This result is not used for selection.

## Data and design

The original retained build contains 801 forecast rows, from 2011-04-07 through
2026-10-08. This is a historical artifact count, not permission to evaluate the
holdout. Development loading admits 608 pre-2023 rows and explicitly purges the
one forecast whose target crosses the boundary; 607 raw WTI/inventory rows
remain. New public-workflow builds decode only history through 2022-12-31.

WTI and Brent come from separately checked ALFRED vintage archives. EIA archived
inventory reports provide commercial crude and Cushing levels, weekly/four-week
changes and report age. Missing values, definition changes, quarantined releases
and late labels remain explicit. The archive supports a qualified reconstruction
of recorded availability; it does not prove exact first-publication timestamps
or immutable first releases.

The chronological replay refits training-only imputation, scaling and models at
each issuance. Active training labels honor revisions and withdrawals. Scoring
uses frozen first jointly available outcomes. Forecasts are made at 18:00 New
York, with the preceding calendar date as the original inclusive source cutoff.
The target calendar is fixed US business days, not exchange trading sessions.

- [Price data audit](docs/data-audit.md)
- [Dataset specification](docs/dataset.md)
- [Inventory source audit](docs/inventory-audit.md)
- [Evaluation protocols and complete experiment history](docs/evaluation.md)
- [Development milestones and original commits](docs/development-history.md)
- [Fresh-checkout and retained-source workflows](docs/reproducibility.md)
- [Research card and frozen-phase boundaries](docs/research-card.md)

Current audit reports under `reports/` omit source-price quotations and mark
those omissions explicitly. Older report excerpts remain in the unchanged Git
history; the omissions were a precaution, not a finding that those particular
excerpts were prohibited. Raw captures, byte-exact backup files, full per-forecast
inputs, predictions and fitted diagnostics remain under ignored `data/`.
CI runs synthetic tests and the offline demonstration without market data.

## License and contributions

Original code, documentation and synthetic fixtures are [MIT licensed](LICENSE).
External data remain subject to [their own terms](DATA_LICENSE.md).
Use the original checkout for development and normal Git pushes; a separate
export or working copy is not required. See the [publication workflow](docs/public-release.md).
See [contributing](CONTRIBUTING.md).
