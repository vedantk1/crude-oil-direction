# Public distribution without private history

The current public quickstart is original MIT code and an invented synthetic
demonstration. Real ALFRED training permission remains unresolved; see
[data rights](../DATA_LICENSE.md) and [the free-source audit](free-training-data-audit.md).
The historical development results remain unchanged and the final holdout stays
excluded from fitting, prediction, scoring and tuning.

## What was preserved and sanitized

Before replacing public content, original report and document bytes were saved
under ignored `data/private-preservation/2026-10-14/`, with checksums. The public
price, vintage and reconciliation reports omit 146 source-price fields across
the three files, including numeric strings and individual price changes. Their
metadata gives original hashes, original byte counts and omission counts.
Introduced nulls mean **source quotation omitted**, not missing source data.
Dates, coverage, release evidence and aggregate findings remain available.
Price quotations in the public source-audit prose were replaced with qualitative
descriptions. Original model reports, predictions, datasets and captures remain
unchanged. Sanitization traversed raw JSON structure and decoded only object
keys, without interpreting price scalars, including any later-history values.

Future low-level source-audit writers automatically preserve raw report bytes
under `data/private-report-archive/<sha256>/` before writing sanitized `reports/`
outputs. Raw output is permitted only under ignored `data/`; different existing
private originals cannot be overwritten. Synthetic tests exercise all three CLI
writers so new audit runs cannot silently restore quotations in public reports.

## Local Git history is private

Sanitizing current files does **not** remove older report excerpts or author
metadata from Git history. That history remains preserved locally. Do not push
this historical repository as the public release, include its `.git/` directory
in a ZIP, or copy ignored captures/private backups into a public repository.
No historical commits or authors were rewritten during this work.

After committing the reviewed source, create a fresh clean export:

```sh
.venv/bin/python scripts/export_public_release.py --output-root data/releases/reviewed
```

Use your compatible pinned environment interpreter if it has another name.
The exporter requires a clean tracked working tree, reads committed HEAD files,
and refuses unsafe paths, symlinks, private/data files or unsanitized source-price
reports. It writes a deterministic ZIP plus a file/checksum manifest into a new
ignored output directory. The archive contains no `.git/`, ignored data,
environments or local author identity metadata. Source commit IDs and file hashes
remain as provenance; they do not include Git author records.

Inspect that export before creating a separate public repository from its
contents. Keep the original local repository and ignored originals private.
Creating, pushing or publishing the public repository is a separate action;
none happened here. Hosted CI still needs its first actual run after publication.
Passing synthetic tests establishes software behavior, not forecasting skill or
source-data permission.

## Presentation

Describe the project as an audited research benchmark with chronological
evaluation and negative results. Report the 301 scored development outcomes,
the simple baselines, retrospective model selection and timing sensitivity.
Do not advertise the highest observed accuracy as a reliable prediction rate,
describe source access as training permission, or claim a trading edge.
