# Chronological development evaluation

The original protocol compares two simple rules with a WTI-only logistic regression. Later experiments add Brent, inventories, and one predefined stronger inventory-model penalty on the same development dates. Each protocol is checkpointed before computing its first results. The underlying price target and availability convention remain those in the [dataset specification](dataset.md).

## Time boundaries and final holdout

| Period | Role |
| --- | --- |
| 2011-04-07 through 2016-12-31 | Initial training history |
| 2017-01-01 through 2022-12-31 | Expanding chronological development replay |
| 2023-01-01 onward | Closed final holdout |

Each development forecast refits using strictly earlier forecast rows whose labels were available at that issuance cutoff. Earlier development outcomes can join later training only once available. There is no random split. The forecast and its exact target date must both precede the holdout boundary; crossing targets are excluded. Scoring uses first-available outcomes known by 2022-12-31. A prediction whose outcome is still unavailable remains explicitly unscored.

The final holdout is excluded from fitting, prediction, scoring, and tuning. The loader checks the full source-file hashes but excludes holdout rows before materializing their feature or label values. Earlier dataset auditing reported full-history coverage, so this is a holdout from model development, not a claim that the underlying data have never been read. There is no CLI switch to run it.

## Fixed models

| Model | Prediction rule | Probability estimate |
| --- | --- | --- |
| Majority | Most frequent class in that issuance's eligible training labels; ties choose up | Training up-class frequency |
| Momentum | Up if the available five-business-day WTI dollar change is positive; zero means not up | None |
| WTI logistic regression | Six declared WTI inputs, regularized logistic regression; probability at least 0.5 means up | Fitted up-class probability |

Missing momentum falls back to that fit's majority prediction and is flagged. It is not assigned an invented probability. Logistic regression uses median imputation and standard scaling fitted only on that training window; an entirely missing training column remains present and is filled with zero. The original WTI model adds neither missingness indicators nor Brent inputs. Parameters are fixed at L2 regularization, `C=1`, `lbfgs`, `max_iter=2000`, no class weights, and `random_state=0`. No parameter or threshold search is performed.

Each fit requires at least 150 eligible training rows and both classes. Insufficient history, invalid inputs, or nonconvergence fail the evaluation rather than silently substituting a different classifier. These rules are recorded in `oil_eval/protocol.py`.

The preprocessing pipeline follows scikit-learn's guidance to fit transforms on training data and apply them to later observations. References: [leakage prevention](https://scikit-learn.org/1.7/common_pitfalls.html), [logistic regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html), [imputer behavior](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html).

## Scoring and interpretation

All models are compared on the same scored development forecast dates. Report accuracy, balanced accuracy, confusion counts, and yearly breakdowns. For majority and logistic probabilities also report ROC AUC, Brier score, and natural-log loss; lower Brier/log loss is better. Log loss clips probabilities to `[1e-15, 1-1e-15]`. Balanced accuracy and AUC are undefined when truth contains only one class. Momentum has no probability metrics.

Training uses active historical label versions, including corrections and withdrawals; scoring intentionally freezes the first jointly available outcome. Later revisions or withdrawals cannot alter that issuance's earlier training data. Events and scoring labels from 2023 onward are excluded, even for older forecast IDs. Source version end dates crossing the holdout boundary are masked in the development view.

The primary comparison is paired development accuracy, with probability quality and yearly stability reported alongside it. A few hundred weekly examples and potentially overlapping horizons do not justify independent-sample significance claims. Development performance is not final holdout performance, trading profitability, or a deployment claim.

## Development results

The protocol and loader were committed as `13489cb` before the first real-data model comparison. The fixed experiment then produced 310 forecasts from **2017-01-06 through 2022-12-22**. Of those, **301 had a first-available outcome by the development scoring cutoff**; nine remain explicitly unscored. One later pre-holdout forecast was excluded because its target crossed into 2023. The 297 initial-history rows are for training rather than development scoring.

| Model | Accuracy | Balanced accuracy | ROC AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-majority baseline | 49.50% | 51.71% | 0.4864 | 0.2512 | 0.6955 |
| Five-day momentum | 50.17% | 49.72% | — | — | — |
| WTI logistic regression | 49.50% | 49.12% | 0.4879 | 0.2578 | 0.7179 |

The same 301 outcomes are used for every row in the table. They contain 168 up and 133 not-up outcomes. Logistic regression and majority each classify 149 correctly; momentum classifies 151 correctly. Against majority, logistic alone is correct on 85 dates and majority alone on another 85. Against momentum, logistic alone is correct on 42 dates and momentum alone on 44. These paired counts do not establish a meaningful gain from logistic regression. Its Brier and log-loss scores are also worse than the historical-prevalence baseline.

| Development year | Scored outcomes | Majority accuracy | Momentum accuracy | Logistic accuracy |
| --- | ---: | ---: | ---: | ---: |
| 2017 | 52 | 44.23% | 51.92% | 42.31% |
| 2018 | 50 | 52.00% | 46.00% | 48.00% |
| 2019 | 50 | 38.00% | 64.00% | 48.00% |
| 2020 | 50 | 50.00% | 56.00% | 54.00% |
| 2021 | 50 | 64.00% | 46.00% | 52.00% |
| 2022 | 49 | 48.98% | 36.73% | 53.06% |

Yearly variation is substantial, and no model is consistently better across these small yearly samples. This result concerns this six-feature model and protocol; it does not prove that all oil-direction prediction is impossible. Parameters were not changed after seeing these results, and the final holdout was not evaluated.

Training windows contained **295–596 eligible labelled examples**. All 310 logistic fits converged, taking at most seven solver iterations. No development forecast needed the missing-momentum fallback. Fits ran on the CPU with one numerical-library thread; no GPU or paid service was used.

## Reproduction and verification

Install the pinned dependencies from `requirements.txt`, then run from the workspace root:

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --report data/evaluations/wti_reproduction.json
.venv/bin/python -m unittest discover -s tests -q
```

The experiment consumes the existing immutable dataset rather than rebuilding it. It writes a content-addressed directory under `data/evaluations/`, containing:

- `predictions.jsonl`: every development prediction, training-input checksum, training date and label-maturity bounds, model probabilities, fitted imputer/scaler/coefficients, and explicit scoring status.
- `summary.json`: aggregate and yearly metrics on paired dates, exclusions, and the closed-holdout status.
- `manifest.json`: source dataset hashes, fixed protocol, evaluation code hashes, pinned runtime versions, and artifact hashes.

The current path and full numerical results are in [the committed report](../reports/development_evaluation.json). `--output-root` and `--report` can choose other workspace-relative destinations; there is no date-range override or holdout execution option. Different code, inputs, or recorded runtime versions create a new run identity. Conflicting bytes at an existing identity cause a failure.

The initial WTI checkpoint passed **99 tests**. They cover holdout filtering before decoding, incompatible source conventions, future-feature/revision invariance, label withdrawals and maturity, target-boundary purging, preprocessing isolation, missing values, baseline tie rules, known metric cases, paired scoring, and artifact protection. A repeated real-data run produced identical prediction, summary, and manifest bytes. Hash and maturity checks passed for all 310 saved forecasts, with zero holdout predictions.

## Fixed Brent comparison

The Brent comparison adds the existing six Brent features plus the aligned Brent-minus-WTI spread and its age, giving fourteen inputs versus the original six WTI inputs. Both logistic regressions retain the same regularization, solver, preprocessing, threshold, and chronological boundaries. No parameter search or final holdout evaluation is authorized by this comparison.

The forecast IDs, dates, targets, scoring availability, and outcome labels must exactly match the original WTI development run. The entire cohort is checked before fitting. If any original forecast lacks eligible Brent inputs, the comparison fails instead of silently dropping dates. Optional missing lookbacks are still imputed within each training window.

Both models use the same ordered historical training rows and active labels at each issuance: the intersection of WTI and combined-input eligibility. Any training rows removed from the original WTI cohort are reported. The WTI training-input hash retains its original six-feature definition. Whenever those hashes match the original run, the implementation must reproduce the original WTI, majority, and momentum outputs exactly, including fitted diagnostics.

The prior development report, its immutable manifest, and both artifacts are verified by hash. The reference must match the source dataset, fixed WTI protocol, and recorded runtime. Original results are preserved; Brent results receive a separate report and run identity. Hash verification reads source bytes, but 2023-onward observations remain excluded from model fitting, prediction, scoring, and tuning.

This comparison protocol and its implementation were committed as `77c24e9` before computing Brent results. The final holdout remains closed until a later explicitly authorized evaluation.

Run the comparison against the retained WTI reference after installing the same pinned runtime:

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment brent-comparison
```

The default reference is `reports/development_evaluation.json`. `--reference-report` may select another authenticated WTI report with the same dataset, protocol, and runtime. The comparison defaults to `reports/brent_comparison.json` and rejects any report destination that would overwrite the original reference or its artifacts. Preserve the original report when rerunning the WTI-only command after code changes: use `--report data/evaluations/wti_reproduction.json` to record that new run separately.

All **141 tests pass** at the Brent protocol checkpoint. Additional synthetic tests cover combined-input validation, unchanged WTI model behavior, shared training rows, missing Brent lookbacks, exact reference dates and outcomes, integrity failures, and rejection before fitting. Future Brent values cannot alter earlier predictions. No real Brent metrics are used to choose the protocol or parameters.

## Brent comparison results

Run `d5c874bd369545258290` uses the original WTI reference `eaf91570f5aaab09a992`. Both regressions predict all **310 original development forecasts**, with exactly **301 scored and nine unscored**. No forecast dates, targets, scoring labels, or availability states changed. Combined eligibility removed **zero historical training rows** at every fit, preserving the original 295–596 training-row range.

| Model | Correct / scored | Accuracy | Balanced accuracy | ROC AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WTI-only logistic | 149 / 301 | 49.50% | 49.12% | 0.4879 | 0.2578 | 0.7179 |
| WTI + Brent logistic | 154 / 301 | 51.16% | 51.71% | 0.4836 | 0.2666 | 0.7336 |

Adding Brent yields **five more correct predictions**, or **+1.66 percentage points** in accuracy. Both models are correct on 116 dates and wrong on 114; Brent alone is correct on 38, while WTI alone is correct on 33. Probability quality worsens: Brier increases by 0.0088 and log loss by 0.0157; AUC falls by 0.0043. The majority and momentum baselines reproduce the original results above. The combined model's probability scores are also worse than the majority baseline's.

| Development year | Scored outcomes | WTI-only accuracy | WTI + Brent accuracy | Additional correct predictions |
| --- | ---: | ---: | ---: | ---: |
| 2017 | 52 | 42.31% | 55.77% | +7 |
| 2018 | 50 | 48.00% | 46.00% | -1 |
| 2019 | 50 | 48.00% | 58.00% | +5 |
| 2020 | 50 | 54.00% | 52.00% | -1 |
| 2021 | 50 | 52.00% | 46.00% | -3 |
| 2022 | 49 | 53.06% | 48.98% | -2 |

The accuracy gain occurs in two of six years; the combined model loses accuracy in the other four. This is a small, inconsistent development gain with worse probability estimates, so it does not establish Brent as a reliable improvement. No settings were adjusted after seeing these results, and neither model has been evaluated on the final holdout.

All 310 six-feature WTI training hashes match the reference, and every original WTI, majority, and momentum payload—including fitted diagnostics—reproduces exactly. Both regressions converged at every issuance: at most seven iterations for WTI and 32 for WTI/Brent. A repeated comparison produced identical prediction, summary, and manifest bytes. Saved hashes and label-maturity bounds passed verification, with no forecasts or targets entering 2023.

The [separate comparison report](../reports/brent_comparison.json) records full aggregate and yearly metrics, paired disagreements, shared-cohort counts, and reference-reproduction checks. Its local manifest additionally records the original report/artifact hashes. The dataset and original WTI report remain unchanged; execution used the existing CPU runtime with no paid data, downloads, or GPU.

## Added constant benchmarks

The following benchmarks were added retrospectively after the WTI and Brent experiments. Both constant direction rules are reported, without choosing a class from development labels. The neutral probability is fixed at 0.5 and uses the existing threshold/tie rule to predict up. Direction-only rules have null probability metrics rather than treating their predictions as certainty.

| Benchmark | Correct / scored | Accuracy | Balanced accuracy | ROC AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always up | 168 / 301 | 55.81% | 50.00% | — | — | — |
| Always not up, including ties | 133 / 301 | 44.19% | 50.00% | — | — | — |
| Constant probability 0.5 | 168 / 301 | 55.81% | 50.00% | 0.5000 | 0.2500 | 0.6931 |

WTI-only logistic scores 19 fewer outcomes correctly than always-up, a 6.31-percentage-point deficit. WTI/Brent scores 14 fewer, a 4.65-point deficit. Both fitted models have worse Brier and log-loss scores than the neutral probability. The constant direction rules' balanced accuracy remains 50%; their differing ordinary accuracies arise from the 168 up and 133 not-up outcomes. These are development diagnostics, with no claim about final holdout performance.

```sh
.venv/bin/python -m oil_eval.benchmarks
```

The command authenticates the existing WTI and Brent reports, run identities, source/runtime agreement, original reference hashes, and all saved prediction artifacts. It independently recomputes the saved aggregate and yearly model metrics before scoring constants. Forecasts, targets, scoring availability, and labels must match between the two runs. Held-out forecasts are rejected before JSON decoding; future outcome availability is rejected before scoring. No model fitting or source-dataset read occurs.

The [supplemental report](../reports/development_benchmarks.json) contains aggregate/yearly benchmark metrics and paired comparisons against both logistic models. Its content-addressed directory under `data/evaluations/` retains compact per-forecast benchmark outputs, the summary, provenance, and hashes. Original reports and immutable artifacts are protected against overwrite. Repeated runs produce identical outputs. The benchmark implementation adds **13 passing synthetic tests**, covering fixed predictions, metric cases, exact cohorts, integrity failures, holdout rejection, and artifact protection. Future replay summaries also include these constants without changing the original model family or frozen protocols.

## Fixed inventory comparison

The inventory experiment adds five columns to the six WTI inputs, retaining the original logistic settings and all development dates. It uses the [audited EIA inventory archives](inventory-audit.md) and retains their qualification: archived values are not proved immutable first releases. The model has only WTI and inventory inputs, with fixed parameters, threshold, and CSV source policy.

| Added input | Definition |
| --- | --- |
| Commercial weekly inventory change | Published within-issue difference, million barrels |
| Commercial four-week inventory change | Current level minus the exact observation 28 calendar days earlier, using the latest version available at issuance with the same definition regime |
| Cushing weekly inventory change | Published within-issue difference, million barrels |
| Cushing four-week inventory change | Same exact-date, historically available comparison for Cushing stocks |
| Inventory report age | Calendar days since actual latest eligible publication, shared because both series use the same report |

Each historical feature row is frozen at its own forecast cutoff, admitting only releases before that date and no later than the preceding calendar day. Four-week references can use current or prior-week columns of an already released issue. Subsequent releases and restatements cannot rewrite earlier training features. Commercial comparisons crossing the October 2016 definition boundary remain null unless both endpoint versions share the same definition.

Before the first retained release, all five columns are null. If the latest declared issue is unusable, all four changes remain null; report age is still known from the release metadata. The July 5, 2019 forecast therefore stays in the cohort with missing changes. No older report is substituted, and the lower-precision PDF remains a possible separate experiment.

Missing inventory never removes a training or forecast row. Values and missing reasons stay explicit in saved feature rows and prediction provenance. Training-only median imputation handles nulls; a column entirely absent in a training window uses zero. No missing-indicator model inputs are added. Each augmented fit records per-column training missing counts and forecast missing columns.

The WTI and inventory models use exactly the same ordered WTI-eligible examples and active historical labels at every fit. All original six-column training hashes, baseline outputs, and WTI model payloads must reproduce exactly; any divergence fails the experiment. Forecast dates, targets, labels, and scored/unscored states are checked against the authenticated WTI reference before fitting. Boundary-crossing rows remain excluded, and inventory features are not constructed for those rows.

The feature loader authenticates the saved inventory report, manifest, all four artifacts, original source snapshot, extracted records, overlaps, and availability summary. The audit must refer to the same authenticated WTI run as the comparison. Historical audit code hashes remain provenance; changed evaluation code does not invalidate the original audit's source evidence.

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment inventory-comparison
```

The default audit is `reports/inventory_audit.json` and reference is `reports/development_evaluation.json`. `--inventory-audit-report` and `--reference-report` select compatible authenticated inputs. The comparison writes a separate `reports/inventory_comparison.json` and content-addressed run under `data/evaluations/`. Its additional `inventory_features.jsonl` stores raw pre-holdout inventory values and missingness provenance for initial training and development; predictions also retain the raw inventory inputs. Original datasets, reports, source captures, and older run artifacts are preserved.

This protocol and implementation were committed as `1ac70a8` before computing inventory model results. Evaluation remains restricted to 2017–2022, with the 2023-onward final holdout closed.

All **247 tests pass** at the inventory protocol checkpoint. New checks cover authenticated audit/source failures, exact four-week lookbacks and revisions, historical feature freezing, definition changes, early missing history, quarantine with known age, training-only imputation, retained rows, strict WTI reproduction, prefit cohort validation, holdout boundaries, and preservation of historical reports/artifacts.

## Inventory comparison results

Run `8660d7d658d2fe7c8762` uses inventory audit `3a01db73789436d8ea29` and the original WTI reference `eaf91570f5aaab09a992`. Both models predict **all 310 original development dates**, with exactly **301 scored and nine unscored**. The original dates, targets, outcomes, availability states, and ordered training cohorts are preserved. Training windows remain 295–596 examples, with zero rows removed for missing inventory.

| Model / benchmark | Correct / scored | Accuracy | Balanced accuracy | ROC AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WTI-only logistic | 149 / 301 | 49.50% | 49.12% | 0.4879 | 0.2578 | 0.7179 |
| WTI + inventory logistic | 158 / 301 | 52.49% | 52.11% | 0.5111 | 0.2574 | 0.7174 |
| Always up | 168 / 301 | 55.81% | 50.00% | — | — | — |
| Constant probability 0.5 | 168 / 301 | 55.81% | 50.00% | 0.5000 | 0.2500 | 0.6931 |

Inventories add **nine correct predictions**, or **+2.99 percentage points** in accuracy. Both regressions are correct on 127 dates and wrong on 121; inventory alone is correct on 31 and WTI alone on 22. Balanced accuracy rises by 2.99 points and AUC by 0.0232. Brier score improves by only 0.0004 and log loss by 0.0005. The inventory model still trails always-up by ten correct predictions and has worse probability scores than the neutral 0.5 benchmark. The original majority and momentum baseline outputs also reproduce exactly.

| Development year | Scored outcomes | WTI-only accuracy | WTI + inventory accuracy | Additional correct predictions |
| --- | ---: | ---: | ---: | ---: |
| 2017 | 52 | 42.31% | 51.92% | +5 |
| 2018 | 50 | 48.00% | 46.00% | -1 |
| 2019 | 50 | 48.00% | 48.00% | 0 |
| 2020 | 50 | 54.00% | 56.00% | +1 |
| 2021 | 50 | 52.00% | 56.00% | +2 |
| 2022 | 49 | 53.06% | 57.14% | +2 |

Accuracy improves in four years, ties in one, and declines in one. Probability quality improves in three years and worsens in three. This is a small development improvement over WTI-only, with little aggregate probability gain and no final holdout validation. No settings changed after these results.

### Explicit missingness

The saved feature artifact contains **607 rows: 297 initial-training opportunities and 310 development forecasts**. The one target crossing the holdout boundary is excluded from inventory construction. Missing counts below describe these frozen raw rows, before training-window imputation:

| Inventory column | Missing initial + development rows | Missing development forecasts |
| --- | ---: | ---: |
| Commercial weekly change | 17 | 1 |
| Commercial four-week change | 23 | 1 |
| Cushing weekly change | 17 | 1 |
| Cushing four-week change | 20 | 1 |
| Shared report age | 16 | 0 |

Sixteen early 2011 rows have no earlier retained release, so all inventory fields remain null. Three further rows lack the exact four-week reference for either series. Three commercial four-week comparisons have incompatible definitions across the October 2016 boundary. The July 5, 2019 forecast has four null stock changes and a known report age of two days; it remains predicted and scored. Once mature, its original label and frozen null inventory inputs also remain eligible for later training.

Raw nulls, missing reasons, source/reference dates and hashes, definition regimes, fitted missing counts, imputer statistics, and forecast missing columns are saved. The augmented model's training-only imputation handles these inputs without dropping examples or rewriting source records.

### Verification and artifacts

All 310 WTI training-input hashes and all original baseline/WTI model payloads reproduce exactly, including fitted diagnostics. Both regressions converged at every issuance: at most seven iterations for WTI and eleven for WTI/inventory. All 247 tests pass. Repeated runs produce identical feature, prediction, summary, and manifest bytes; independent metrics, historical feature reconstruction, source/provenance hashes, and label-maturity checks verify the saved outputs.

The [inventory comparison report](../reports/inventory_comparison.json) contains aggregate/yearly metrics, paired disagreements, constant benchmarks, missingness counts, and reference reproduction. The immutable local run additionally contains all 607 raw inventory rows and per-forecast provenance. Original datasets, audits, reports, and earlier runs remain preserved. Execution used the existing CPU environment without downloads, paid services, or a GPU, and the final holdout remains closed.

## Fixed inventory regularization experiment

This retrospective development experiment changes only the WTI + inventory logistic model's L2 penalty from `C=1` to **`C=0.1`**. The candidate was chosen after reviewing the earlier development results, whose probability scores were worse than neutral and historical-prevalence probabilities. It is a model-selection experiment, not a previously unseen validation result. There is no grid, threshold search, feature change, new data capture, or holdout execution.

The new `logistic_wti_inventory_c01` model retains the same eleven columns, ordered training examples and labels, expanding fit schedule, training-only median imputation and scaling, solver, iteration limit, class weights, and 0.5 decision threshold. The original WTI and inventory `C=1` regressions and both historical rules are replayed as controls. All fixed constant benchmarks are scored on the same available outcomes.

Before fitting, the runner authenticates compatible WTI, inventory-audit, and `C=1` inventory-comparison references. Both selected model references must record the same native runtime fingerprint as the candidate's environment. Newly reconstructed raw inventory rows, including initial-history nulls and provenance, must have exactly the same bytes/hash as the reference's frozen feature artifact. Development dates, targets, scoring states and raw forecast inventory inputs must match the references. After replay, every reference control's complete training metadata, raw inputs/provenance, predictions, probabilities and fitted diagnostics must reproduce exactly. A mismatch stops the run. Historical runs without native fingerprints remain valid historical evidence, but this experiment requires refreshed controls; disclose historical numerical differences and never relax the exact checks.

```sh
.venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment inventory-regularization
```

Defaults are `reports/development_evaluation.json`, `reports/inventory_audit.json`, and `reports/inventory_comparison.json`. Override them with `--reference-report`, `--inventory-audit-report`, and `--inventory-reference-report` only for compatible authenticated runs. This command writes a separate `reports/inventory_regularization.json` and immutable local run; the original reports, dataset, audit and run artifacts are protected from replacement. These commands require the retained local evidence, as do the earlier comparisons.

### Compatible runtime and refreshed controls

The existing SciPy 1.15.3 macOS 14 arm64 wheel failed locally while importing PROPACK: its thread-local zero-fill section was rejected by the native loader. Installed binary hashes matched their package RECORD. An isolated, checksum-verified official macOS 12 arm64 wheel of **the same SciPy version** imports successfully, but its different numerical build cannot reproduce the saved historical controls bit-for-bit. The original installation and all historical artifacts remain preserved; no binary patch or relaxed comparison check is used. This local observation does not establish a universal wheel defect.

The alternative wheel is available from [official PyPI files](https://pypi.org/project/scipy/1.15.3/#files), named `scipy-1.15.3-cp313-cp313-macosx_12_0_arm64.whl`, with SHA256 `7e11270a000969409d37ed399585ee530b9ef6aa99d50c019de4cb01e8e54e62`. Install it into ignored `.audit-deps/scipy-macos12` with `pip install --no-deps --target` after verifying its hash. Other pinned packages and Python remain unchanged. Prefix the commands below with `PYTHONPATH=.audit-deps/scipy-macos12` to select this isolated dependency.

Establish matched controls using the retained evidence, without downloading data or replacing published reports:

```sh
PYTHONPATH=.audit-deps/scipy-macos12 .venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --report data/regularization/wti-reference.json
PYTHONPATH=.audit-deps/scipy-macos12 .venv/bin/python -m oil_audit.inventory \
  --snapshot-dir data/audit/inventories/20261013T075852032262Z \
  --reference-report data/regularization/wti-reference.json \
  --evidence-manifest data/audit/inventory-evidence/20261013T081104639545Z/manifest.json \
  --probe-manifest data/audit/inventory-probes/20261013T081558994391Z/manifest.json \
  --quarantine-manifest data/audit/inventory-anomalies/20261013T082020828938Z/manifest.json \
  --report data/regularization/inventory-audit-reference.json
PYTHONPATH=.audit-deps/scipy-macos12 .venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment inventory-comparison \
  --reference-report data/regularization/wti-reference.json \
  --inventory-audit-report data/regularization/inventory-audit-reference.json \
  --report data/regularization/inventory-comparison-reference.json
PYTHONPATH=.audit-deps/scipy-macos12 .venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment inventory-regularization \
  --reference-report data/regularization/wti-reference.json \
  --inventory-audit-report data/regularization/inventory-audit-reference.json \
  --inventory-reference-report data/regularization/inventory-comparison-reference.json
```

The final pair changes only `C` within this common working runtime. Separately compare refreshed `C=1` controls with the historical run for exact cohorts, labels, training metadata, raw inventory features/provenance and class predictions; disclose probability/diagnostic differences from the runtime build. Python/package version strings alone do not guarantee bitwise reproducibility across native wheels. New run identities therefore also hash NumPy, SciPy and scikit-learn wheel metadata and native libraries, using package-relative names rather than installation paths.

The saved [runtime control check](../reports/regularization_runtime_check.json) verifies all 310 cohorts, labels, complete training records, raw inventory/provenance and both rule baselines exactly against historical run `8660d7d658d2fe7c8762`. The full frozen inventory artifact is byte-identical, and neither logistic model changes any class prediction. Maximum probability differences are `8.88e-16` for WTI and `1.31e-13` for inventory; median differences are zero. Accuracy, balanced accuracy and AUC are unchanged. Inventory Brier/log-loss differences are below `2e-15`. These are disclosed numerical differences, not a claim of historical bitwise reproduction. The refreshed WTI run is `95f93cf564c9cc55ede7`, inventory audit `090d8516fd820fbb1d34`, and `C=1` inventory run `e908d9c3437095815a70`.

All **285 tests pass** before the regularization result, covering native-build identities, strict selected-reference fingerprints, authenticated controls, null inputs, training-only preprocessing, paired metrics, prefit schedule failures, holdout rejection and historical artifact protection.

## Inventory regularization results

The protocol, implementation, tests and historical runtime check were committed as `5a90916` before computing the candidate. Run `21e67f2c5ee9e4c216d6` compares `C=0.1` with the refreshed `C=1` inventory reference `e908d9c3437095815a70`. Both selected references have exactly the candidate's native runtime fingerprint. All **310 development forecasts, 301 scored outcomes, nine unscored states, 295–596-example training windows, and 607 frozen inventory rows** are unchanged.

| Model / benchmark | Correct / scored | Accuracy | Balanced accuracy | ROC AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WTI-only logistic | 149 / 301 | 49.50% | 49.12% | 0.4879 | 0.2578 | 0.7179 |
| WTI + inventory, C=1 | 158 / 301 | 52.49% | 52.11% | 0.5111 | 0.2574 | 0.7174 |
| WTI + inventory, C=0.1 | 160 / 301 | 53.16% | 52.87% | 0.5077 | 0.2563 | 0.7126 |
| Training-majority / prevalence | 149 / 301 | 49.50% | 51.71% | 0.4864 | 0.2512 | 0.6955 |
| Five-day momentum | 151 / 301 | 50.17% | 49.72% | — | — | — |
| Always up | 168 / 301 | 55.81% | 50.00% | — | — | — |
| Always not up | 133 / 301 | 44.19% | 50.00% | — | — | — |
| Constant probability 0.5 | 168 / 301 | 55.81% | 50.00% | 0.5000 | 0.2500 | 0.6931 |

Relative to `C=1`, the candidate adds **two correct predictions (+0.66 percentage points)**. Both are correct on 155 dates and wrong on 138; the candidate alone is correct on five and the control alone on three. Balanced accuracy improves by 0.75 points. Brier decreases by 0.00108 and log loss by 0.00483, while aggregate AUC falls by 0.00336. Lower probability loss does not imply better ranking discrimination.

| Year | Scored | C=1 accuracy | C=0.1 accuracy | Additional correct | Brier difference C=0.1 minus C=1 | Log-loss difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 52 | 51.92% | 51.92% | 0 | -0.00048 | -0.00119 |
| 2018 | 50 | 46.00% | 50.00% | +2 | -0.00028 | -0.00067 |
| 2019 | 50 | 48.00% | 50.00% | +1 | -0.00274 | -0.00577 |
| 2020 | 50 | 56.00% | 54.00% | -1 | -0.00178 | -0.01834 |
| 2021 | 50 | 56.00% | 56.00% | 0 | -0.00028 | -0.00065 |
| 2022 | 49 | 57.14% | 57.14% | 0 | -0.00092 | -0.00247 |

Both probability losses improve in **all six years**, with the largest log-loss change in 2020. Accuracy improves in two years, worsens in one and ties in three. This supports a modest development improvement in probability quality from the stronger penalty, but the candidate still trails neutral and historical-prevalence probabilities, and trails always-up by eight correct predictions. It does not establish a useful forecasting edge, significance, profitability or final-holdout performance. No further settings were changed after observing this result.

All selected controls reproduce exactly across 310 forecasts, including complete training records, raw inventory values/provenance, probabilities and diagnostics. Missingness remains unchanged: sixteen early rows without history, exact-lookback/definition gaps, and the quarantined forecast remain explicit. All three regressions converge, taking at most seven WTI, eleven `C=1` inventory and nine `C=0.1` iterations. The [regularization report](../reports/inventory_regularization.json) records aggregate/yearly metrics, paired comparisons, constant benchmarks and strict reference checks. Its immutable run also retains raw inputs and fitted diagnostics. The final holdout remains closed.

Independent reconstruction verifies aggregate/yearly metrics, benchmark scores, all control payloads, artifact hashes, all 27 evaluation/audit code hashes against `5a90916`, and all 211 native-file hashes. Repeating the run produces byte-identical report, features, predictions, summary and manifest. A checksum comparison confirms all **696 pre-existing data/report files** remain unchanged. The timed repeat plus artifact-preservation checks took **13.13 seconds**, excluding imports; this includes source authentication, feature construction, 930 logistic fits, scoring and file checks, rather than measuring fitting alone. All 285 tests pass in this working runtime. No paid data, GPU or new source-data capture was used.

## Fixed rolling-window comparison

Compare the expanding `C=0.1` inventory model with **one four-calendar-year rolling policy**, keeping all 310 development dates and the 301 first-available scored outcomes fixed. The window is chosen before computing predictions from a capacity-only check: three years yields 147–155 eligible examples and violates the unchanged 150-row minimum on 119 dates; four years yields 197–207 examples and both classes on every date. No accuracy or probability score informed this choice, and no window-duration search is performed.

At each issuance, select the ordered, historically eligible expanding training rows first. The rolling model then retains only examples with forecast origins in `[issuance minus four calendar years, issuance)`, clamping February 29 to February 28 if the lower year has no leap day. Label maturity, current historical label state, and target-before-origin checks apply before this filtering. Do not reach farther back to replace withdrawn labels, drop rows with missing inventory, or move the lower boundary to meet a count. Insufficient history or a missing class fails the run.

Both inventory variants use the same eleven inputs, `C=0.1`, solver and 0.5 threshold. Imputation and scaling are fitted separately on each model's own training rows; inventory inputs remain frozen at each example's original issuance. Fewer samples at the same `C` also change the penalty relative to average training loss. This tests the practical fixed-`C` rolling policy, including the smaller sample and refitted preprocessing; it does not isolate market recency alone.

All five prior expanding control models are replayed and must reproduce the selected regularization reference exactly. The runner authenticates the WTI, inventory audit, `C=1` and expanding `C=0.1` reference chain, its identical frozen inventory artifact, and the common native runtime before fitting. Per-forecast `training` metadata retains the expanding run's exact shape and values; separate `rolling_training` metadata records the fixed bounds, count/up count, actual retained origin range, label-maturity bounds, WTI/inventory hashes, excluded old-row count, and inventory missing counts. Original datasets, source captures, reports and run artifacts are preserved.

Use the existing documented working runtime and retained references:

```sh
PYTHONPATH=.audit-deps/scipy-macos12 .venv/bin/python -m oil_eval \
  --dataset-dir data/datasets/4abada098156b2b39306 \
  --experiment inventory-window-comparison \
  --reference-report data/regularization/wti-reference.json \
  --inventory-audit-report data/regularization/inventory-audit-reference.json \
  --inventory-reference-report data/regularization/inventory-comparison-reference.json \
  --regularization-reference-report reports/inventory_regularization.json
```

This writes a separate `reports/inventory_window_comparison.json` and content-addressed immutable run. `--regularization-reference-report` selects a compatible authenticated expanding `C=0.1` run. The default historical WTI/`C=1` references lack native fingerprints; use the explicit compatible references above or recreate the chain with the [runtime recipe](#compatible-runtime-and-refreshed-controls).

Assess paired accuracy, balanced accuracy, AUC, Brier/log loss, yearly differences and the simple benchmarks on the same scored cohort. This is another retrospective development choice after earlier experiments, with growing model-selection exposure; it is not unseen validation. Report the result even if it deteriorates. No parameters, window duration, features, thresholds or dates may be changed in response to these results. The 2023-onward final holdout remains closed.

All **314 tests pass** before computing rolling predictions. New checks cover calendar bounds, leap-day clamping, current/future-origin rejection, retained order and nulls, rolling-only preprocessing, active delayed/withdrawn labels, separate metadata/hashes, minimum-history/class failures without fallback, reference/native authentication, unchanged expanding controls, matched scored/unscored states, holdout boundaries, and historical artifact protection. The paired-score calculation is shared with the regularization comparison.

### Data quality and predictive limits

The observations are audited rather than assumed clean: historical versions, publication gaps, stale inventory bytes, definition changes and null inputs are explicitly recorded. That supports a qualified historical replay, with remaining first-release and exact-intraday availability uncertainty. It does not establish that the inputs contain a useful forecasting signal.

Only 301 weekly development outcomes are scored, with potentially dependent horizons. Delayed spot benchmarks and a narrow eleven-input feature set also constrain the question. Inventory changes are measured levels/differences, not surprises relative to contemporary market expectations. The weak fitted results do not prove that the underlying prices or stock observations are bad; they show that this limited information set and model family have not demonstrated an edge. The rolling experiment tests a different history policy, not a repair of source quality.

Assess paired accuracy, balanced accuracy, AUC, Brier score and log loss, plus yearly stability. The main hypothesis is that stronger regularization improves probability quality relative to `C=1`; improvement beyond neutral and historical-prevalence probabilities would be more useful evidence. A gain caused only by approaching constant probabilities is not a discovered forecasting edge. No settings may be changed in response to this experiment's results, and no independent-sample significance claim is warranted. The 2023-onward final holdout stays closed.
