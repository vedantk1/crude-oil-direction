# Historical direction dataset

Built from the free ALFRED archives captured on 2026-10-13. This reconstructs **recorded archive states under a conservative timing convention**. The [source audit](data-audit.md) explains why it cannot prove exact intraday availability or availability through every public channel.

## Implemented forecast definition

| Item | Rule |
| --- | --- |
| Target benchmark | WTI spot, `DCOILWTICO` |
| Additional benchmark | Brent spot, `DCOILBRENTEU` |
| Start | Requested from 2011; first archive is 2011-04-06, so first forecast is 2011-04-07 |
| Forecast schedule | Next fixed US business day after each listed WTI vintage; merge updates that map to the same date |
| Issuance time | 18:00 America/New_York, with timezone offsets recorded per row |
| Information cutoff | Preceding **calendar** date, inclusive; Monday normally uses Sunday |
| Horizon | Fifth fixed US business day after issuance date, excluding issuance date |
| Direction label | 1 if WTI on target date exceeds WTI on issuance date; otherwise 0, including ties |
| Missing endpoint | Unavailable label; do not shift to another observation date |

The origin for the label is the issuance date's WTI benchmark observation, which may only be published later. It is **not** the older price in the feature vector. This measures benchmark direction between two date-labelled prices; it does not simulate an executable trade at 18:00. Spot benchmarks are not exchange-traded contracts.

The selected calendar is Monday–Friday excluding recurring US federal observed holidays, with Juneteenth from 2021. Saturday holidays are observed Friday and Sunday holidays Monday, including adjacent-year New Year observance. Good Friday, regional Inauguration Day, and exceptional closures are not excluded. This is an operational calendar, not NYMEX sessions or a claim about EIA's publication calendar. The implementation supports 2011–2100; later years project those recurring rules. Sources: [OPM holiday rules](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/), [Juneteenth implementation](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/juneteenth-national-independence-day-holiday.pdf).

## Availability and source bounds

WTI and Brent are reconstructed independently. A price must have an active ALFRED interval at the cutoff, a finite numeric value, and an observation date no later than that cutoff. Any version whose archive start predates its observation is permanently quarantined. An active missing revision cannot be replaced by an older numeric version. Overlapping or noncontiguous adjacent intervals cause a failure.

Matching observation and vintage dates are allowed because issuance occurs on a subsequent business day. This differs intentionally from the older audit helper's stricter query-date behavior. Original interval ends remain in source evidence and label lookup logic; they are never model inputs.

For example, the 2020-04-23 forecast uses WTI through **April 21**, Brent through **April 13**, and a spread dated **April 13**. The respective age features are two, ten, and ten calendar days. A join on each series' latest row would produce a misleading spread here.

Each series also has an independent source-capture bound. Forecast issuance and label eligibility must fall within both retained source captures. Timestamped files use their recorded UTC completion times. Where the manifest records only a date without a timezone, the builder uses 00:00 UTC on the preceding calendar date as a deliberately early lower bound on capture time. It does not replace the recorded provenance or assert an actual retrieval hour. The earliest file bound per series governs the build. This prevents a download from materializing tomorrow's forecast or future label eligibility.

The replay stops at the latest scheduled issuance supported by those bounds. For this snapshot that is **2026-10-08 at 18:00 New York**, with information through October 7. No current rows were excluded by the capture guard. Training exports beyond that replay date are rejected; data are not silently extended to today.

## Feature columns

The WTI-only input set has six columns. The combined set has fourteen: six each for WTI and Brent, plus the spread and its age. Only these allowlisted columns enter a training export.

| Per-benchmark feature | Meaning |
| --- | --- |
| `latest_price` | Latest usable observed price, USD per barrel |
| `age_calendar_days` | Issuance date minus that observation date |
| `change_5_business_days_usd` | Latest price minus price exactly five calendar-policy business days before that observation |
| `change_20_business_days_usd` | Same definition over twenty business days |
| `change_std_20_business_days_usd` | Population standard deviation of twenty consecutive dollar changes, requiring all twenty-one exact prices |
| `coverage_20_business_days` | Fraction of those twenty-one exact observations available, between 0 and 1 |

Column names are prefixed `wti_` or `brent_`. `brent_wti_spread_usd` is Brent minus WTI on their latest common usable observation date; `spread_age_calendar_days` is that date's age at issuance.

Windows are anchored to each series' own latest available observation. If that observation falls on a federal holiday, it remains the anchor and the preceding dates follow the selected calendar. Missing endpoints yield null changes, and any missing window observation yields null volatility. No nearest-date substitution, forward filling, smoothing, scaling, imputation, or log transformation is applied. Dollar changes and direct comparisons remain defined for negative WTI prices. Historical context outside the calendar's supported range retains price and age but has null window features.

All issuance rows are retained. `eligibility.wti` requires a usable WTI latest price; `eligibility.wti_brent` also requires Brent and a common-date spread. Separate `complete_*_features` flags identify fully populated vectors. Eligibility does not imply a known label or a complete window, and it imposes no hidden staleness threshold. Optional missing inputs must be handled within future training folds.

## Labels and training cutoffs

`label_events.jsonl` records each outcome state, including initially missing endpoints, numeric corrections, and later withdrawals. An endpoint change becomes eligible on the next fixed US business day after its vintage date. Multiple changes mapping to the same day are resolved using the preceding calendar day's archive state. `available_from` is inclusive; `available_until` is exclusive. A null end is open only within the manifest's bounded replay period.

At each simulated training date, use the active event and retain it only if `label_status` is `available`. Both endpoint prices must be available under the cutoff rule. Training can therefore lose an example following a withdrawal or change its class after a correction. The CLI performs this selection; selecting every eventual label would leak information into earlier training windows.

For primary evaluation, `evaluation_labels.jsonl` freezes the **first jointly available numeric endpoint pair**. It records when that outcome became usable and its endpoint vintage starts. Later corrections do not rewrite this scoring file. This is an explicit scoring policy, not a claim that initial prices are economically final. The file is for evaluation only; it must not replace the active historical training-label lookup. Comparing first-available scoring with revised scoring can be a later robustness check. Evaluation should also honor the outcome's availability date.

## Saved artifacts and measured coverage

The current directory is recorded in [the dataset summary](../reports/dataset_summary.json). Artifacts are local JSON Lines and JSON under ignored `data/datasets/`:

| File | Contents |
| --- | --- |
| `features.jsonl` | Stable row ID, issuance/cutoff/target dates, numeric feature dictionary, input version references, eligibility flags |
| `label_events.jsonl` | Versioned outcome states keyed by row ID, separate from features |
| `evaluation_labels.jsonl` | Frozen first-available outcomes and their availability dates |
| `manifest.json` | Configuration, feature allowlists, source URLs/hashes/capture provenance, code hashes, artifact hashes, and summary |

The directory ID derives from source-manifest, configuration, and code hashes. Rebuilding identical inputs is idempotent. An existing artifact with different bytes is not overwritten. `training-view` checks artifact hashes before selection. Data files remain local; see the source audit for vendor-origin data reuse limits.

| Measured result | Count |
| --- | ---: |
| Forecast opportunities, 2011-04-07 to 2026-10-08 | 801 |
| Label-state events | 2,422 |
| Outcomes that became available for first-available scoring | 789 |
| Active available labels at replay end | 777 |
| Previously available outcomes now unavailable | 12 |
| Outcomes whose class changed between available versions | 1 |
| Usable latest-price inputs for either feature set | 801 |
| Complete WTI feature vectors | 694 |
| Complete combined feature vectors | 568 |

At replay end, the other 24 label states comprise six missing targets, eleven missing origins, six missing both endpoints, and one quarantined endpoint. These categories include the two forecasts whose target date had not yet passed the cutoff; missingness is not all permanent. The 789 first-available outcomes contain 413 up and 376 not-up labels, with zero ties. These counts do not measure predictive performance.

Median latest-data age is three calendar days for each benchmark. Maximum age is nine days for WTI and ten for Brent. Twenty-day volatility is null in 107 WTI rows and 199 Brent rows; the US operational calendar does not guarantee a reported spot price on every selected date. The summary gives every feature's missing count.

## Verification and next experiment

The dataset checkpoint passed **55 tests**, including future-revision invariance, non-resurrection after withdrawal, permanent quarantine, exact-date lookbacks, negative prices, holiday boundaries, weekend updates, label maturity and expiry, feature allowlisting, capture bounds, checksum failures, and deterministic rebuilds. The subsequent [evaluation implementation](evaluation.md) brings the complete passing suite to 99 tests. The full saved dataset build also passed date-order checks across all 801 rows, 33,253 numeric window references, and all 2,422 label events. The source audit separately validated the underlying archives against independent exports and current EIA prices.

The dataset artifacts contain no fitted model or preprocessing. The separate [development evaluation](evaluation.md) now implements expanding chronological training with historical label availability, training-only preprocessing, two baselines, and a WTI-only logistic regression. Its final holdout starts in 2023 and remains closed. Brent comparison, additional issuance delay, and scoring-policy sensitivity remain future experiments; the current builder implements the one-business-day convention only.

Build and historical training-view commands are in the [README](../README.md).
