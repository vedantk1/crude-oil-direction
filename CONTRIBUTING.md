# Contributing

Crude Oil Direction Prediction's initial research phase is frozen. The project is a
forecasting benchmark with negative and mixed findings, not a validated price
or trading predictor.
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
remain subject to [their own terms](DATA_LICENSE.md). Real ALFRED training-use
permission is unresolved; use synthetic fixtures for contributions and do not
start new capture/training runs. Alternative permitted sources require a
separate availability audit and protocol; see [the source audit](docs/free-training-data-audit.md).

Tests and CI use synthetic data without market downloads or keys. A passing demo
proves pipeline behavior, not live source availability or forecasting skill.
Current source-audit reports omit quotations; never interpret their redacted
nulls as source missingness. Older report excerpts remain in Git history. Keep
raw captures and full artifacts under ignored `data/`, and use the original
repository for development, commits and pushes. No separate release checkout is
required; see the [publication workflow](docs/public-release.md).
Use a new output root for every source replay. Record capture identity,
environment and whether a result uses retained snapshots or newly captured
history.

Do not use held-out values for development, tuning, diagnostics or examples.
Future history may be hashed as raw bytes for integrity, but cannot be decoded
as numerical input by the development workflow. Any future holdout evaluation
requires a separately agreed final model, protocol and explicit authorization.
