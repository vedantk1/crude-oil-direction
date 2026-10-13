# Chronological development evaluation

The original protocol compares two simple rules with a WTI-only logistic regression. The fixed Brent extension below compares six WTI inputs with fourteen combined inputs on the same development dates. Each protocol is checkpointed before computing its first results. The underlying price target and availability convention remain those in the [dataset specification](dataset.md).

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
