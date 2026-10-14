# Free data for a future training phase

Review date: **2026-10-14**. This is a source-permission and metadata review,
not a new experiment. No source datasets were downloaded, no models were fitted,
and the project's 2023-onward final holdout was not opened. Existing data and
results remain evidence of the completed research phase.

**There are credible free training sources, but this review found no verified
replacement for the exact daily WTI/Brent series plus ALFRED historical vintages.**
Permission to use data and evidence of historical availability are separate
requirements. A replacement must satisfy both; changing the market, frequency,
label or availability clock starts a different experiment.

## Evidence matrix

“Supports training” below means an affirmative broad reuse grant or open license
supports this use. It does not mean the publisher has specifically certified
this project or that every linked third-party item has the same rights.

| Direct source | Permission evidence and assessment | Frequency and history | Historical availability and fit |
| --- | --- | --- | --- |
| **EIA commercial-crude and Cushing inventories** | EIA permits use and distribution of its own government data. These are EIA survey statistics, unlike the licensed spot prices in the same report. **Supports training on the published inventory aggregates**, with attribution and publication date; protected third-party items remain excluded. | Weekly; the existing project audit covers 2011–2022, with early-2011 gaps and a commercial-stock definition break. | Dated WPSR issue archives are already audited here. They provide publication evidence, not proof of immutable first releases. Useful features, or a separate inventory-direction target; they do not supply a cleared WTI price target. [Reuse](https://www.eia.gov/about/copyrights_reuse.php), [weekly survey methodology](https://www.eia.gov/petroleum/supply/weekly/pdf/appendixb.pdf), [project archive audit](inventory-audit.md). |
| **World Bank Commodity Prices — History and Projections / Pink Sheet** | The **dataset-specific catalog entry explicitly licenses the dataset under CC BY 4.0**. This is affirmative permission evidence, not an inference from free access. **Supports training under that license**, subject to attribution, the Bank's added dataset terms and any specifically identified third-party exceptions. | Official monthly-history publication identifies monthly WTI and Brent columns; history as a whole begins in 1960, but individual series start later and have missing observations. The catalog's 1960–2030 range combines history and projections; its “Annual” periodicity field does not describe the linked monthly workbook. | Catalog says updates occur on the second business day of the month. The report archive has dated historical issues, but does **not establish a complete monthly revision archive** for 2011–2022. Strong candidate for a **separate monthly crude-oil direction benchmark**, after checking the selected workbook's notices and archive coverage. [Dataset license](https://datacatalog.worldbank.org/search/dataset/0038238/commodity-prices-history-and-projections), [monthly-series metadata](https://thedocs.worldbank.org/en/doc/386771467756369668-0050022016/render/CMOHistoricalDataMonthly.pdf), [archive index](https://www.worldbank.org/en/research/commodity-markets/report-archive). |
| **EIA U.S. domestic crude-oil first-purchase price** | The table's source is government survey **Form EIA-182**, rather than Refinitiv/LSEG spot quotes. EIA's own-data reuse grant therefore **supports training on this published survey aggregate**. Attribute the source and release date. | Monthly U.S. series begins in 1974; EIA-182 supplies the modern period. It measures the first purchase at the lease/wellhead, **not the daily Cushing WTI benchmark**. | The official Petroleum Marketing Monthly archive lists 2011 issues and their observation endpoints, typically about three months behind publication. Current values are preliminary/revised before finalization. A release-by-release reconstruction still needs validation. Attractive alternative if government-created crude-price statistics matter more than keeping the WTI target. [Definitions and source](https://www.eia.gov/dnav/pet/TblDefs/pet_pri_dfp1_tbldef2.asp), [series metadata](https://www.eia.gov/dnav/pet/pet_pri_dfp1_k_m.htm), [monthly issue archive](https://www.eia.gov/petroleum/marketing/monthly/archive/). |
| **EIA weekly U.S. retail gasoline or on-highway diesel estimates** | Published national/regional estimates are EIA survey outputs, covered by its own-data reuse grant. **Supports training on the published aggregates**; this does not grant access to or rights over confidential respondent data or commercial inputs used in sampling/imputation. | Weekly; regular gasoline history covers 2011 onward. Gasoline's May 2018 and diesel's June 2022 sample changes create comparability breaks. A different energy-price target, not crude spot direction. | WPSR Table 14 is an archive candidate, but its complete 2011–2022 history was not audited here. Current methodology says Tuesday around 10:00 ET; older official notes say Monday around 17:00 ET. Verify date-specific publication evidence rather than applying today's schedule retrospectively. [Survey sources and historical notes](https://www.eia.gov/dnav/pet/tbldefs/pet_pri_gnd_tbldef2.asp), [gasoline methodology](https://www.eia.gov/petroleum/gasdiesel/gas_proc-methods.php), [diesel methodology](https://www.eia.gov/petroleum/gasdiesel/diesel_proc-methods.php), [WPSR table index](https://www.eia.gov/petroleum/supply/weekly/index.php). |
| **ECB euro foreign-exchange reference rates, EXR** | ECB permits free use of information obtained directly from its website, with accurate source attribution, disclosure of modifications and additional conditions for sold documents. **Broad reuse supports a research-training use**; no training-specific prohibition was found in that permission. Select ECB reference rates, not the separate Bloomberg-sourced FX dataset. | Daily USD/EUR reference series starts 1999-01-04, covering the project's development era. Published on TARGET operating days, a different calendar from fixed US business days. | Current framework specifies publication around 16:00 CET and allows correction until the next TARGET business day's rate is published. This does not prove the entire historical revision timeline. A one-TARGET-day safety margin is a candidate convention requiring an audit. This is a **separate FX benchmark**, not an oil-data substitute. [ECB reuse terms](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html), [series metadata](https://data.ecb.europa.eu/data/concepts/united-states-dollar?layerType=KL&tags_array%5B0%5D=United+States+dollar), [reference-rate framework](https://www.ecb.europa.eu/stats/pdf/exchange/Frameworkfortheeuroforeignexchangereferencerates.en.pdf). |

## Existing price-source restrictions

The [FRED legal page](https://fred.stlouisfed.org/legal/) currently states in
**Summarized Terms, II. Prohibited Use** that FRED services/content cannot be used
for software or machine-learning development/training. That section expressly
includes noncommercial, educational and personal use. **Full Terms,
Introduction and Applicability** includes ALFRED. The **FRED API Restrictions (k)**
also explicitly bars training.

The general Full Terms prohibition list does not repeat the summary's AI bullet.
That difference does not establish an ALFRED exemption. No written clarification
or permission was obtained, and this review does not establish when the wording
changed. Fresh ALFRED capture and real-data training should not be advertised as
cleared public workflows while permission remains unresolved. Synthetic fixtures
and original MIT code are separate from those source rights.

Fetching the same spot prices directly from EIA would remove the FRED access
route, but would **not establish training permission for the underlying data**.
EIA's [spot-price source notes](https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp)
name Refinitiv, an LSEG business. Its
[forecasting methodology](https://www.eia.gov/analysis/handbook/pdf/STEO_Crude_Oil_Price.pdf)
says EIA purchases WTI and Brent history and republishes it by permission.
EIA's general government-data permission has protected-material exceptions;
this audit found no explicit training grant for those vendor-origin spot series.

## What the World Bank license establishes

The Bank's [dataset terms](https://www.worldbank.org/ext/en/legal/terms-conditions/datasets)
grant free dataset access, default to CC BY 4.0 unless specifically labeled
otherwise, and require attribution to the Bank and known data providers. They
also retain third-party exceptions and an additional dispute-resolution term.
The commodity dataset's own CC BY 4.0 label is stronger evidence than the general
default. Do not relabel downloaded data MIT or assume a license for other
commercial providers' original feeds.

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) allows copying and
adaptation for any purpose with attribution, a license link and change notices.
Creative Commons' [AI guidance](https://creativecommons.org/faq/index.html)
explains that licensed reuse includes new technologies, subject to the license
conditions and other applicable rights. This supports using the specifically
licensed Pink Sheet dataset for training. **Workbook-specific clearance remains
conditional:** the selected monthly workbook's internal source/license notices
were not inspected. A bounded attempt to resolve its official download link
failed first under restricted networking, then returned HTTP 403 from the
commodity-markets landing page when retried with network access. No workbook
bytes or numerical cells were read, and no capture/hash was created. The official
catalog license and linked older monthly-series metadata were verified through
the web reader; they do not establish whether the current workbook carries a
more specific exception or names additional underlying providers. Retain those
notices and resolve that final metadata check before implementing the data path.

## Recommendation

Publish the project's original code and synthetic demonstration with the
completed findings framed as a historical research benchmark. Retain the
ALFRED-derived originals privately and document the unresolved permission; do
not start new ALFRED training or describe an unverified source switch as a fix.

For a later real-data phase, the first choice is **monthly WTI direction from the
World Bank's licensed Pink Sheet**, if keeping the WTI subject is the priority.
The alternative with the clearest government-survey provenance is **monthly
EIA U.S. first-purchase-price direction**. Either needs a bounded archive audit
and a separate written protocol before training. Neither result would be
comparable to the existing five-business-day experiment.

There are only **144 calendar months in 2011–2022**, before missing values,
lookbacks and label maturity. The 72 months before the existing 2017 development
start cannot meet the current 150-row training minimum. A monthly phase would
need justified earlier training history or a newly specified split and minimum;
it should use a very small CPU model and probability baselines. No GPU or paid
data is required for that scope.

The new audit must record observation period, actual release date, vintage,
retrieval date, missingness and source terms separately. A current revised
spreadsheet plus an assumed lag cannot recreate historical information. Exclude
post-2022 observations before numerical decoding, preserve the original target
and holdout artifacts, and obtain separate authorization before any final-holdout
evaluation. No replacement dataset or revised model result exists yet.
