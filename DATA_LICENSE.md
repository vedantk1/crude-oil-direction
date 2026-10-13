# Source data and reuse

The [MIT license](LICENSE) covers original project code, documentation and the
invented synthetic demonstration. It does not grant rights to external data,
third-party libraries or their licenses.

Downloaded data and full model artifacts remain local in ignored `data/`.
Public reports contain aggregate research results and source identities.
Contributions must not include downloaded prices, source archives or credentials.

- **WTI and Brent:** ALFRED supplies recorded vintages of EIA spot-price series.
  EIA attributes these prices to Refinitiv/LSEG and notes republication by
  permission. Free access does not establish unrestricted redistribution rights.
  Use the download tools to obtain your own copies and review the current
  [EIA price source notes](https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp),
  [FRED legal terms](https://fred.stlouisfed.org/legal/) and
  [EIA reuse policy](https://www.eia.gov/about/copyrights_reuse.php).
- **EIA inventories:** EIA generally permits reuse of its own government
  material, with exceptions for protected third-party material. Attribute the
  source and capture date. This repository keeps captured release files local
  rather than asserting a blanket license for every archived file.
- **Synthetic demo:** generated arithmetic values and fixtures are original
  project material. They can be redistributed under MIT; their scores say
  nothing about market forecasting ability.

The [data audit](docs/data-audit.md) and [inventory audit](docs/inventory-audit.md)
record provenance and historical-availability limitations separately from rights.
