# EIA inventory archive audit

This audit concerns free historical inventory information for **2011–2022**, before the final model holdout. It captures EIA issue archives, checks their release and observation dates, parses the two inventory series, and measures their availability on the existing development forecast dates. It adds no inventory features to the price dataset and fits no model.

The completed audit admits **596 of 597 intended issues** and covers **309 of the 310 original development forecast dates**. One stale July 2019 CSV is quarantined. Early 2011 publication coverage remains incomplete, and unchanged first-release values are not proved. Full results are in the [inventory audit report](../reports/inventory_audit.json).

## Measured results

| Check | Result |
| --- | --- |
| Verified release span | 2011-07-27 through 2022-12-29 |
| Declared current observation span | 2011-07-22 through 2022-12-23 |
| Intended issues | 597: 596 indexed and one unlisted supplement |
| Usable Table 4 issues | 596 for each series |
| Missing declared Friday weeks inside that span | 0 |
| Unusable current-week CSV | Week ending 2019-06-28, nominal release 2019-07-03 |
| Original development dates checked | 310, with the original dates and targets preserved |
| Latest stock, weekly change, compatible four-week history available | 309 dates for each series |
| Latest issue unusable at forecast | 2019-07-05 |
| Observation age on covered forecasts | 6–10 calendar days, median 6 |
| Publication age on covered forecasts | 1–3 calendar days, median 1 |
| Repeated-observation comparisons | 594 per series, 1,188 total |
| Observed commercial definition adjustment | 1 |
| Other observed changes in those overlaps | 0 for either series |
| Independent Table 1 commercial comparisons | All three sampled issues match |

Release counts are 23 in 2011; 53 each in 2014 and 2020; and 52 in every other year from 2012 through 2022. All required series rows and date schemas passed for admitted tables. Commercial published changes differ from rounded-level subtraction in 44 issues, always by at most 0.001 million barrels. No Cushing rounding residual was observed.

The overlap at the definition boundary confirms why regimes matter: the September 30, 2016 commercial stock is 499.740 million barrels in the October 5 release, then 469.108 in the October 13 release's prior-week column. The 30.632-million-barrel difference is classified as the definition adjustment, rather than an ordinary weekly draw. Finding no other overlap changes cannot establish that older files were never corrected in place.

The coverage check deliberately requires the latest declared issue to be usable. It flags the July 5, 2019 forecast rather than silently substituting an older issue. This is an audit of availability, not a decision to drop that date from a future model comparison. A later feature experiment must retain the original dates and explicitly handle unavailable or separately recovered inputs.

## Sources and coverage

The [official Weekly Petroleum Status Report archive](https://www.eia.gov/petroleum/supply/weekly/archive/) is the primary release-date authority. Each entry distinguishes publication date from the observation week ending. Table 4 CSVs contain both `Commercial (Excluding SPR)` and `Cushing`, in million barrels, and download without a paid service or API key.

The captured index links **596 issues** during 2011–2022. Two November 2016/2017 entries sit in an HTML row missing its opening tag; the parser recovers them and independently checks that every bounded issue link was retained. The capture also includes the [July 27, 2011 issue](https://www.eia.gov/petroleum/supply/weekly/archive/2011/2011_07_27/wpsr_2011_07_27.html), whose official page confirms its release date although the index omits it. These produce 597 intended releases.

The index's 2011 listings begin August 3. The July 27 supplement extends verified publication coverage by one week. Bounded probes for April 6, May 4, June 1, July 6, and July 20 returned missing resources at the tested official locations. This is a gap in the retained evidence, not proof that no earlier official archive exists. Early 2011 inventory history must remain missing until independently verified; later archives must not be backfilled into an earlier forecast.

Current historical API/download series are useful for other checks but do not replace issue archives when reconstructing historical availability. In particular, EIA backcast the commercial-inventory definition in 2016.

## Release timing

Use the displayed publication date, with inventory information admitted only when `release_date < forecast_date`. This follows the existing 18:00 New York issuance and preceding-calendar-day cutoff. Earliest eligible issuance is the next fixed US business day after publication. Observation age and publication age are different quantities.

Holiday and disruption dates come from the archive, rather than a rule that every Friday observation was published the following Wednesday. Examples:

- The week ending October 11, 2013 was released October 21.
- Data for both June 17 and June 24, 2022 were released June 29 after a hardware failure. The older issue lives in a folder named `2022_06_17_data`; that name is not a publication date. [EIA's planned delayed release](https://www.eia.gov/pressroom/releases/press506.php), [publication confirmation](https://www.eia.gov/pressroom/releases/press507.php).

The [2011 report methodology](https://www.eia.gov/petroleum/supply/weekly/archive/2011/2011_08_03/pdf/appendixb.pdf) describes the usual Wednesday release of CSV/XLS at 10:30 a.m. Eastern, with other report formats later. This supports a conservative next-day convention. It does not establish exact historical intraday delivery for every file.

## Inventory definitions and parsing

The parser selects exact row labels and validated date columns rather than fixed row numbers. It supports UTF-8 and legacy Windows-1252. Repeated `Percent Change` headers require a positional CSV reader. Required stock values must be finite and nonnegative; current and previous observation dates must be consecutive Fridays, preceding actual publication.

Published differences use unrounded inputs while displayed stock levels have three decimals. A discrepancy of up to 0.00101 million barrels is allowed between the published difference and subtraction of the displayed levels. Both numbers and their residual are retained. Required missing values, unexpected schemas, duplicate rows, mismatched dates, and checksum failures are explicit failures.

Table 1's upper stock section supplies an independent commercial-total comparison in three sampled issues. Its lower `Commercial Stock Change` row uses thousand barrels per day and must not be substituted for a weekly stock change.

Beginning with the **October 13, 2016 release**, commercial stocks exclude oil held on leases, as well as the Strategic Petroleum Reserve. Earlier archived commercial levels include lease stocks. The series labels did not change, and EIA published a backcast history. [Dated EIA definition notice](https://www.eia.gov/todayinenergy/detail.php?id=28292)

Preserve a commercial-definition regime for each release. The audit separates repeated-observation changes crossing this definition boundary from other changes in the archives. A four-week comparison must use compatible definitions. Around the boundary, subtracting independently published levels can create an artificial draw; the issue's internally comparable weekly change is the relevant starting point.

## Evidence limits and next use

These are values **as preserved in EIA's issue archive at capture time**. Checksums preserve this capture; they do not prove that every archived file is an immutable original. Comparing repeated observations across releases can detect some changes, but cannot reveal every correction made directly to an older file. HTTP modification times are retained for provenance and do not establish historical publication or revision dates.

The next model experiment should use a small, fixed inventory feature set only after accepting this qualification. Candidate inputs are the published weekly change, a compatible four-week change, and report age. A stock build or draw is not a consensus surprise: that would require separately verified historical expectations. The final holdout remains closed.

## July 2019 quarantine and possible fallback

The [July 3, 2019 Table 4 CSV](https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/csv/table4.csv) contains June 21/June 14 headers rather than the required June 28/June 21. Its bytes exactly match the preceding June 26 issue. The July 3 Table 1 CSV is stale too, so it does not repair that issue.

The dated [issue page](https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/wpsr_2019_07_03.php) links a [correct Table 4 PDF](https://www.eia.gov/petroleum/supply/weekly/archive/2019/2019_07_03/pdf/table4.pdf). Visual inspection confirms June 28/June 21 headers and both requested stock rows, but the PDF prints levels and differences to only **0.1 million barrels**. Its commercial weekly change is -1.1 million barrels and Cushing's is +0.7 million barrels.

A future PDF fallback could recover this forecast's inventory inputs with an explicit precision field and separate provenance. It has not been admitted to the automatic CSV audit or used in a model. July 10's CSV later reports more precise June 28 levels; those must not be assigned to July 3 availability. The stale CSV bytes are retained and authenticated separately as quarantine evidence. Correct PDF bytes and the dated issue page are also retained as fallback evidence under ignored `data/audit/inventory-fallback-evidence/`.

## Reproduction

Run from the workspace root with the existing pinned Python environment:

```sh
.venv/bin/python -m oil_audit.inventory_sources
```

This captures a new timestamped snapshot under ignored `data/audit/inventories/`. Downloads are bounded to release years 2011–2022, with twelve workers and a shared maximum of four request starts per second. The index contains other years' date metadata; inventory tables outside the audit window are never requested. Original source bytes are preserved, and manifests retain hashes, URLs, retrieval metadata, and failed attempts.

To resume the retained capture after an interruption or failed request:

```sh
.venv/bin/python -m oil_audit.inventory_sources \
  --resume data/audit/inventories/20261013T075852032262Z
```

Resumption verifies saved bytes before using them, retries missing or failed captures, and rejects changed bytes instead of overwriting them. A completed capture means every intended issue was attempted; the separate success flag and offline audit establish whether all tables were valid.

Reproduce the completed audit offline:

```sh
.venv/bin/python -m oil_audit.inventory \
  --snapshot-dir data/audit/inventories/20261013T075852032262Z \
  --evidence-manifest data/audit/inventory-evidence/20261013T081104639545Z/manifest.json \
  --probe-manifest data/audit/inventory-probes/20261013T081558994391Z/manifest.json \
  --quarantine-manifest data/audit/inventory-anomalies/20261013T082020828938Z/manifest.json
.venv/bin/python -m unittest discover -s tests -q
```

The audit authenticates the saved WTI development reference; it reads no source price dataset or final holdout rows. All admitted release dates and all coverage forecast/target dates precede 2023. The report points to `data/audits/inventories/3a01db73789436d8ea29/`, containing extracted inventory records, overlap changes, per-forecast availability, summary, and manifest. Source, auxiliary/probe/quarantine evidence, code, and output hashes are retained. Current and older audit artifacts and original model reports are protected against report-path overwrites.

All **203 project tests pass**, including **49 inventory tests** for schema/date checks, immutable recovery, malformed-index completeness, delayed releases, checksum failures, missing inputs, definition transitions, historical joins, quarantine, and output protection. A repeated audit produced identical report and artifact bytes; independent hash, code-provenance, and date-bound checks passed. The original dataset, WTI report, and Brent report remain unchanged.
