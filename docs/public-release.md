# Publishing Crude Oil Direction

Use the original repository and full development history for development and
publication. The repository name is `crude-oil-direction`, with the project title
**Crude Oil Direction**. Downloaded data, private backups, environments and full
model artifacts stay local; a second working repository is not required.

The public quickstart uses original MIT code and an invented synthetic demo.
Real ALFRED training permission remains unresolved; see [data rights](../DATA_LICENSE.md)
and [the free-source audit](free-training-data-audit.md). The retained development
results remain unchanged. The 2023-onward final holdout stays excluded from
fitting, prediction, scoring and tuning.

## Ordinary development workflow

Make code and documentation changes in this checkout. Review the staged diff
and tracked paths before committing, then push the intended branch to its
GitHub remote. Keep raw source captures, backup files, generated datasets,
predictions and local environments ignored. Do not force-add ignored data.
CI installs the pinned dependencies, runs the unit suite and runs the synthetic
demo without downloading market data or evaluating the final holdout.

The original development commits are part of the project, including experiments
that failed to improve results. Their chronology and hashes are summarized in
[development history](development-history.md). There is no separate sanitized
commit history to maintain. Older commits also preserve the report excerpts and
Git metadata present when they were created.

`scripts/export_public_release.py` remains an optional source-archive utility.
It is not required for GitHub publication or ordinary development. It exports
committed files without `.git/` or ignored data, checks current public report
format, and writes a deterministic ZIP and manifest under a new ignored output
root. That archive has no development history; use Git when history matters.

## Current report format and private artifacts

Before current reports were shortened, original report and document bytes were
saved under ignored `data/private-preservation/2026-10-14/`, with checksums. The
current price, vintage and reconciliation reports omit 146 source-price fields
across three files, including numeric strings and individual price changes.
Their metadata records original hashes, byte counts and omission counts.
Introduced nulls mean **source quotation omitted**, not missing source data.
Dates, coverage, release evidence and aggregate findings remain available.

This omission was a conservative publication precaution. It did not establish
that publishing those particular excerpts was prohibited. The unchanged Git
history retains the older excerpts. Keeping that history does not publish the
ignored raw datasets or grant new rights to external data.

Low-level source-audit writers preserve raw report bytes under ignored
`data/private-report-archive/<sha256>/` before writing concise `reports/` outputs.
Different existing private originals cannot be overwritten. Original datasets,
predictions, full model artifacts and source captures remain local. No model
refit or holdout evaluation is needed to publish the code and its history.

## Presentation

Describe Crude Oil Direction as a forecasting project with chronological
evaluation and a documented development history. Report the 301 scored
development outcomes, simple baselines, retrospective model selection and timing
sensitivity. The strongest selected model reached 53.16% direction accuracy;
always-up reached 55.81%, and simple probability baselines remained stronger.
No useful predictive advantage or profitable strategy has been established.

The project's research findings belong in its documentation. The project name
does not need to imply that a successful predictor has been demonstrated.

## Publication chronology

At checkpoint `74b2213`, a source-only export contained 108 tracked files plus
its generated manifest. Fresh bootstrap, 420 tests and the synthetic demo
through tree comparison passed. A preservation inventory verified 1,469
unchanged existing source/data/result files plus three byte-exact private
originals of the replaced reports. Twelve report/document backup files were
retained. No real-data model was refitted. The
[release-preparation report](../reports/release_preparation.json) records that
checkpoint, its archive checksums and the limitations known then. It describes
the earlier unpublished state.

On 2026-10-14, an export from checkpoint `4308407` was published as a separate
repository named `crude-oil-direction-research`. Its root commit was
`bb02f332c8d48f4bbead2a330b6c946085b81f2f`, with no original development parents;
six documentation files were adapted to the separate publication context. Its
first hosted Offline checks run `38531360671` passed on Ubuntu/Python 3.13.16:
420 tests in 14.213 seconds and the synthetic demo. A second publication commit,
`686c82084c61812a835b3cd4d4364feacdeb4703`, recorded that result.

That snapshot-only arrangement hid the project's development progression on
GitHub and introduced an unnecessary second working repository. The subsequent
decision is to publish from the original repository with all original commits,
use the title Crude Oil Direction, and remove the extra working copy. The
[publication report](../reports/publication.json) records execution and CI
status. Earlier export and publication records remain historical evidence, not
instructions to repeat the separate-copy workflow.
