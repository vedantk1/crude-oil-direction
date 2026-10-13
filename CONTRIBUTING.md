# Contributing

This research phase is frozen. The project is an audited forecasting benchmark
with negative and mixed findings, not a validated price or trading predictor.
The 2023-onward final holdout remains closed. Changes to model selection need a
new documented hypothesis and protocol before any results are computed.

Run commands from the repository root using Python 3.13:

```sh
python3.13 scripts/bootstrap.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m oil_eval.demo --run-root data/contributor-demo
```

On Windows, use `python` for a Python 3.13 interpreter and
`.venv\Scripts\python.exe` for the environment interpreter. The bootstrap refuses
to modify an existing environment; select a new ignored directory with
`--environment` if necessary.

Keep code modular and update the relevant docs with behavior changes. Use
workspace-relative paths, explicit missing values, source checksums and bounded
date admission before value decoding. Never commit credentials, personal or
machine identifiers, downloaded data, environments or full model artifacts.
Original code, docs and synthetic fixtures use [MIT](LICENSE); external data
remain subject to [their own terms](DATA_LICENSE.md).

Tests and CI use synthetic data without market downloads or keys. A passing demo
proves pipeline behavior, not live source availability or forecasting skill.
Preserve historical reports and immutable artifacts; use a new output root for
every source replay. Record capture identity, environment and whether a result
uses retained snapshots or a newly captured history.

Do not use held-out values for development, tuning, diagnostics or examples.
Future history may be hashed as raw bytes for integrity, but cannot be decoded
as numerical input by the development workflow. Any future holdout evaluation
requires a separately agreed final model, protocol and explicit authorization.
