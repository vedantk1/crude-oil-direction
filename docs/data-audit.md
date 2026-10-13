# WTI and Brent data audit

Audit date: 2026-10-13. The free sources are sufficient for a qualified, reproducible forecasting experiment. ALFRED allows reconstruction of its recorded historical states from **2011-04-06 onward** for both series. Exact historical intraday availability, or the earliest price available through every public channel, is not established.

The original proposal to evaluate forecasts back to 2000 needs narrowing for the primary experiment. Observation history starts much earlier than archived availability history. Today's price download plus a fixed delay is demonstrably different from the information in past vintages.

This audit downloaded and inspected the data, compared independent exports, and saved source hashes. No model was trained, no data were purchased, and nothing was published.

## Current price histories

Public EIA XLS downloads succeeded without registration, an API key, or payment. The snapshot contains:

| Measure | WTI | Brent |
| --- | ---: | ---: |
| First numeric observation | 1986-01-02 | 1987-05-20 |
| Last numeric observation | 2026-10-06 | 2026-10-06 |
| Numeric observations | 9,517 | 9,102 |
| Numeric observations from 2000 onward | 6,332 | 6,481 |
| Duplicate dates | 0 | 0 |
| Explicit empty price rows in XLS | 0 | 0 |
| Weekdays without a numeric price within each series' coverage | 1,117 | 1,173 |
| Weekend observations | 0 | 0 |
| Nonpositive prices | 1 | 0 |

The missing weekday counts include holidays and unclassified omissions; they are not counts of missing trading sessions. Missing dates are omitted from the XLS rows, so a check for blank cells alone would miss them. No missing prices were filled and no extremes were deleted.

Within the shared 1987-05-20 to 2026-10-06 period, 8,410 dates have both prices, 791 have only WTI, 692 have only Brent, and 382 weekdays have neither. A simple inner join would discard real WTI observations and could silently change a five-observation target into a different horizon.

WTI is -36.98 USD per barrel on 2020-04-20. The 2020-04-29 archived EIA table independently confirms this. Ordinary log returns fail there, and percentage-return signs can misrepresent direction across negative prices. A future direction label should compare price levels directly, with a separately defined reporting calendar.

Sources: [WTI history and XLS](https://www.eia.gov/dnav/pet/hist/RWTCD.htm), [Brent history and XLS](https://www.eia.gov/dnav/pet/hist/RBRTED.htm). Full calculations, annual counts, gaps, and provenance: [price audit JSON](../reports/price_audit.json).

## Publication timing differs by channel

The earlier statement that EIA publishes these prices only weekly was too broad.

| Channel | Verified evidence | What it establishes |
| --- | --- | --- |
| EIA historical spot table | Release date 2026-10-07; next release 2026-10-15; prices through October 6 | Current history follows its displayed release cycle |
| EIA Today in Energy daily prices | Page says weekday updates excluding federal holidays, typically 7:30–8:30 a.m.; captured page dated October 9 contains October 8 close | A faster free snapshot route exists; no complete historical snapshot archive was verified |
| EIA Weekly Petroleum Status Report | Currently indexed archives begin 2011-08-03; sampled Table 11 files retain daily prices | Dated weekly source snapshots, with source-specific coverage |
| ALFRED | Dated versions for both series from 2011-04-06 | Recorded date-level archive states, including revisions and missing values |

The daily page does not specify a timezone beside its update hours. The WPSR has a documented Eastern-time schedule: ordinarily CSV/XLS tables after Wednesday 10:30 a.m., with PDF/HTML later and holiday exceptions. A scheduled time does not prove the actual arrival time of every historical file. Different sources can carry different subsets on the same day.

ALFRED says its updates generally follow a release within one business day. Its release dates can originate from the source, a provider, or first FRED availability. Therefore an archive date is not a uniform website retrieval timestamp. Adding one business day can be a conservative experiment convention, but it is still an assumption to disclose and test.

Sources: [historical table](https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm), [daily snapshot](https://www.eia.gov/todayinenergy/prices.php), [WPSR archive](https://www.eia.gov/petroleum/supply/weekly/archive/), [release schedule](https://www.eia.gov/petroleum/supply/weekly/schedule.php), [format-specific schedule](https://www.eia.gov/about/new/), [ALFRED methodology](https://alfred.stlouisfed.org/help).

## Historical versions that were actually downloaded

| Measure | WTI | Brent |
| --- | ---: | ---: |
| ALFRED identifier | `DCOILWTICO` | `DCOILBRENTEU` |
| First listed vintage | 2011-04-06 | 2011-04-06 |
| Last listed vintage | 2026-10-07 | 2026-10-07 |
| Listed vintage dates downloaded | 801 | 803 |
| Distinct interval records | 11,470 | 11,236 |
| Observation dates ever numeric | 10,286 | 10,021 |
| Observation dates with multiple distinct numeric prices | 30 | 23 |

The free public form supports downloads without credentials. Two batches per series cover the entire listed vintage set. Each interval records the observation date, price, and inclusive start/end dates for that version. A blank end date means the current version. A blank value is a missing-value revision; it must not be replaced by an older value during reconstruction.

Separate exports for four historical dates were compared against reconstruction from the complete intervals. Every value in all eight series/date snapshot columns matched. All adjacent version intervals are contiguous and nonoverlapping. The current numeric date/value dictionaries also match the independently downloaded EIA XLS histories exactly: **zero mismatches in either series**.

| Requested archive date | Latest numeric WTI observation | Latest numeric Brent observation |
| --- | --- | --- |
| 2011-04-06 | 2011-04-05 | 2011-04-05 |
| 2011-04-13 | 2011-04-12 | 2011-04-12 |
| 2020-04-22 | 2020-04-21 | 2020-04-13 |
| 2020-04-29 | 2020-04-27 | 2020-04-27 |

The April 2020 difference demonstrates why WTI and Brent need independent availability joins. The median first-numeric archive lag, excluding the initial bulk archive load, is six calendar days for each series. There are substantial exceptions; this is a measured archive characteristic, not a fixed source publication delay. For example, WTI's 2024-05-31 observation first appears numerically on 2024-07-10. The audit does not establish that it was unavailable everywhere during those 40 days.

Sources: [WTI download form](https://alfred.stlouisfed.org/series/downloaddata?seid=DCOILWTICO), [Brent download form](https://alfred.stlouisfed.org/series/downloaddata?seid=DCOILBRENTEU), [download format definitions](https://alfred.stlouisfed.org/help/downloaddata). Exact events and checks: [vintage audit JSON](../reports/vintage_audit.json).

## Material revisions and anomalies

**Numeric corrections are visible.** WTI's 2020-03-18 price changes from 26.96 to 20.48 on the April 1 vintage, back to 26.96 on April 22, then to 20.48 on April 29. These are recorded archive differences; their cause is not established. A present-day history with a delay cannot reproduce that sequence.

**Large sets of older values later became missing.** The 2026-09-16 vintage changes 743 WTI and 890 Brent observations from numeric to missing. Of these, 378 WTI and 311 Brent dates are from 2000 onward; 53 WTI and one Brent date are from 2011-04-06 onward. Their cause remains unresolved. Preserve both earlier values and later missing states instead of rewriting history.

An independent EIA archive check confirms one concrete omission: Table 11 released on 2011-08-03 contains WTI **99.53 on 2011-07-22**, while the current EIA XLS omits that date. This is not merely an inferred holiday gap.

**One impossible availability date occurs in each archive.** Both mark a 2012-12-27 observation as active on 2012-12-26. The strict audit lookup permanently quarantines those version records. Six WTI and seven Brent version records have matching observation and vintage dates; that alone is not an error, but the archive cannot prove intraday ordering. The strict lookup excludes same-day observations at the requested cutoff.

## Independent weekly archive checks

Four original EIA Table 11 CSVs were downloaded and checked against the current XLS. All numeric values and missing states in these sampled daily sections agree, except the omitted WTI observation above.

| Release | Daily period in table | Useful evidence |
| --- | --- | --- |
| 2011-08-03 | July 18–29, 2011 | WTI July 22 value omitted from current history |
| 2020-04-22 | April 6–17, 2020 | Does not contain April 20; ALFRED WTI already does on this archive date |
| 2020-04-29 | April 13–24, 2020 | Confirms April 20 WTI -36.98 and Brent 17.36 |
| 2024-01-04 | December 18–29, 2023 | December 26 has WTI 75.84 but no Brent value |

These CSVs use CP1252 encoding, unavailable-value markers, and multiple header sections. In the daily section, the first two date columns are weekly averages, followed by ten daily columns. The audit parser separates them and checks weekday names and year-boundary dates.

Four issues do not prove completeness or immutability of every weekly archive. One inspected archive page also had conflicting metadata: the October 7, 2026 issue path/index opened a header naming September 30, while its price table contained values through October 2. The source index, issue path, header, and table need to be checked together.

Sources: [2011-08-03 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2011/2011_08_03/csv/table11.csv), [2020-04-22 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2020/2020_04_22/csv/table11.csv), [2020-04-29 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2020/2020_04_29/csv/table11.csv), [2024-01-04 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2024/2024_01_04/csv/table11.csv), [inconsistent archive header](https://www.eia.gov/petroleum/supply/weekly/archive/2026/2026_10_07/wpsr_2026_10_07.php). Results: [source reconciliation JSON](../reports/source_reconciliation.json).

## Recommended experiment boundary

1. Use ALFRED's recorded versions as the primary reproducible information source. Restrict claims about historical availability to 2011-04-06 onward. Older observations can be historical context after that date; they do not prove pre-2011 forecast-time availability.
2. Store observation date, version start/end, retrieval time or date, source URL, and checksum separately. Preserve every downloaded version and missing-value transition.
3. Select WTI and Brent independently as of the forecast cutoff. Include data age. Calculate the spread only from mutually available observations for the same observation date.
4. Specify an issuance convention with a conservative delay, and test sensitivity to an additional business day. Do not describe this as exact intraday historical availability.
5. Keep impossible-date records quarantined. Do not silently carry a deleted price forward, use the current series to patch earlier features, or convert missing dates into zero changes.
6. Define the five-day target calendar separately from row positions in a sparse or inner-joined dataset. The implemented dataset now uses five fixed US business days, as selected for the project, and keeps exact endpoints even when a price is missing. See the dataset specification for the scoring policy.
7. Enforce training-label availability as well as feature availability. An outcome is usable in training only after its required endpoint prices were available. Explicitly choose initial versus revised prices for final scoring.

A weekly update experiment has roughly 800 historical update opportunities from this archive, before lookbacks and missing labels. Daily forecasts would often reuse the same information and should not be counted as independent new examples. Small models remain appropriate. The audit establishes data feasibility, not predictive accuracy or a trading edge.

The subsequent [dataset implementation](dataset.md) applies these decisions and records 801 forecast opportunities. It documents the final issuance convention, independent WTI/Brent joins, fixed US business-day horizon, source-capture limits, and separate historical training and evaluation labels.

The separate [2011–2022 inventory audit](inventory-audit.md) examines free EIA commercial-crude and Cushing stock archives for a possible later feature experiment. It uses publication dates independently of observation weeks, preserves the October 2016 commercial-definition break, and qualifies archived values rather than claiming immutable first releases. Its scope ends before the model holdout; it does not alter this price audit or the ALFRED dataset.

## Cost, reuse, and saved evidence

All downloads were free and key-free. EIA also offers a free registered API, but this audit did not require it. The spot data are attributed to Refinitiv/LSEG, and the daily snapshot says they are republished with permission. EIA permits broad reuse of its own material but identifies protected third-party exceptions. Consequently, this repository keeps downloaded prices local under ignored `data/` and provides source download code. It does not assert a CC0 or unrestricted redistribution license for vendor-origin prices. Sources: [EIA Open Data](https://www.eia.gov/opendata/), [price source notes](https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp), [reuse policy](https://www.eia.gov/about/copyrights_reuse.php).

Evidence locations are `data/audit/current/20261013T055923606961Z/`, `data/audit/vintages/`, and `data/audit/releases/`. Their manifests record source URLs, SHA-256 hashes, and retrieval provenance. Initial ALFRED captures have retrieval dates only; current EIA and release captures have UTC completion timestamps. These are this audit's retrieval times, not original historical publication times.

Verification passed: source checksums, current XLS schema/coverage checks, ten unit tests, all interval adjacency checks, eight complete snapshot-column comparisons, full current EIA-versus-ALFRED reconciliation, and four weekly issue comparisons. See [reproducibility](reproducibility.md) for bounded offline replay and free new-download commands. Original code/docs now use [MIT](../LICENSE); external source rights remain [separate](../DATA_LICENSE.md).
