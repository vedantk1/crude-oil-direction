# Source data and reuse

The [MIT license](LICENSE) covers original project code, documentation and the
invented synthetic demonstration. It does not grant rights to external data,
third-party libraries or their licenses. Free access is separate from permission
to train models, redistribute data or reconstruct historical publication.

## Current public scope

The public quickstart and CI use invented synthetic sources only. Historical
ALFRED/EIA development results are preserved as research findings. **No new
ALFRED model training is authorized by a free download or by this repository.**

The current [FRED legal page](https://fred.stlouisfed.org/legal/) includes an
explicit ML development/training restriction in its summarized prohibited uses;
its applicability names ALFRED. The full API restrictions repeat it. The general
full prohibition list does not repeat that bullet; this inconsistency does not
establish an exemption. No written training-use clarification or permission was
obtained. Do not run the real-data capture/training recipes while that use is
unresolved. This does not alter MIT for original code or establish when the
published wording changed.

## Price sources and alternatives

- **Original daily WTI/Brent:** EIA's [spot-price source notes](https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp)
  name Refinitiv/LSEG. Downloading the same quotes directly from EIA would not
  independently establish training permission for the underlying vendor data.
  [EIA reuse](https://www.eia.gov/about/copyrights_reuse.php) has protected-material
  exceptions. These daily spot series remain unresolved.
- **EIA's own inventory and price survey aggregates:** the government-data reuse
  grant supports use of the published aggregates, with source/publication-date
  acknowledgment. Commercial respondent data and protected third-party items
  do not inherit that permission. A survey-based price target changes the task.
- **World Bank Pink Sheet:** its commodity dataset has an explicit CC BY 4.0
  label supporting training with attribution, license/change notices and the
  Bank's applicable dataset conditions. The selected workbook's notices and
  historical releases still need checking before implementation. Monthly
  WTI/Brent are a different horizon from the completed daily experiment.
- **Synthetic demo:** generated values and fixtures are original project material
  under MIT. Their scores have no market meaning.

See the dated [free-training-data audit](docs/free-training-data-audit.md) for
primary-source evidence, exceptions, availability gaps and proposed next steps.
No replacement price dataset has been built or evaluated.

## Public reports and private originals

The three source-price audit reports are public sanitized copies. Their
`public_sanitization` metadata records the original byte count/hash and omitted
field counts. Introduced nulls mean **quotation omitted**, not a missing source
observation. Dates, coverage, findings and aggregate metrics remain available.
Byte-exact originals and all raw captures stay under ignored `data/`.

Audit report writers preserve raw bytes privately before public sanitization.
Do not commit raw captures, price excerpts, full prediction artifacts or
credentials. Local Git history deliberately remains unchanged and contains
older excerpts; publish the [clean release export](docs/public-release.md),
not this private historical repository or a zip containing its `.git/` directory.
