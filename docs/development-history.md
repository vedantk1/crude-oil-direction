# Development history

Crude Oil Direction began as a free, CPU-only project to predict crude-oil price
direction. The task became WTI spot direction over five fixed US business days,
with source information restricted to what the archives record as available at
each forecast. The initial training period is 2011–2016, development is
2017–2022, and the 2023-onward final holdout remains closed.

This is the original development history, not a reconstructed sequence. The
commits below remain unchanged. Experiment definitions, cohort counts, results
and limitations are documented in [evaluation](evaluation.md); the table records
how the implementation and decisions developed.

| Commit | Milestone and outcome |
| --- | --- |
| `336f672` | Built the point-in-time oil direction dataset after auditing WTI and Brent availability separately. |
| `13489cb` | Locked chronological development dates, label availability and the closed final holdout. |
| `fa19ecd` | Compared baselines with WTI-only logistic regression: 49.50% direction accuracy. |
| `77c24e9` | Locked the matched-date Brent feature comparison before computing its results. |
| `53f09a5` | Recorded Brent at 51.16% accuracy: five more correct directions, but worse probability quality. |
| `63af93b` | Added constant benchmarks; always-up achieved 55.81%, exposing the limits of accuracy alone. |
| `b2ed3ca` | Audited free EIA inventory archives for 2011–2022, recording gaps, definition changes and one stale release. |
| `1ac70a8` | Locked weekly/four-week inventory changes, report age and explicit missing-data handling. |
| `15a0673` | Recorded WTI + inventory at 52.49%, nine more correct directions than WTI-only. |
| `5a90916` | Locked the matched C=0.1 regularization comparison and runtime controls. |
| `4d45a3f` | Recorded C=0.1 at 53.16%, two more correct directions than C=1, still behind simple benchmarks. |
| `931d8bf` | Locked a four-year rolling training window against the expanding C=0.1 model. |
| `7635178` | Recorded rolling-window accuracy of 50.83% and worse probability scores. |
| `66d0eeb` | Locked a small boosted-tree comparison on the same WTI and inventory inputs. |
| `5155c9e` | Recorded trees at 50.17%; some probability scores improved, but simple probabilities remained stronger. |
| `f4ca7fa` | Removed dependence on old local run IDs; added fresh bootstrap, workflow, synthetic demonstration and CI configuration. |
| `b84a738` | Locked one extra business day of information and training-label delay, keeping forecast dates and scoring targets fixed. |
| `f7dc55e` | Recorded reproducibility checks and the delay result; froze the phase. Delay accuracy was 53.82%, with 106 of 301 scored directions changed. |
| `2f15a07` | Fixed delay routing so generic evaluation cannot silently run the dedicated delay experiment incorrectly. |
| `74b2213` | Shortened current public audit excerpts, retained originals locally and documented source-use restrictions and alternatives. |
| `4308407` | Recorded clean-export installation/tests and preservation verification without changing original data or earlier commits. |

## Decisions retained

All model comparisons use the same development dates and 301 scored outcomes
from 310 forecasts. The dataset contains explicit unavailable and unscored
states. Training transforms and models are refitted using only eligible
historical information; scoring targets stay frozen. The archive reconstruction
has documented timing limitations and is not proof of exact first publication.

Brent, inventories and stronger regularization produced modest improvements
over WTI-only. Rolling windows and trees did not improve direction accuracy over
expanding C=0.1. No fitted model demonstrated an advantage over simple probability
benchmarks. The delay check was a robustness test, not another model-selection
candidate. Results from repeated development experiments are not independent
confirmation of predictive skill.

The public quickstart uses synthetic data. New ALFRED training remains unresolved
under the source-use terms documented in [data rights](../DATA_LICENSE.md).
Alternative free sources were investigated but no replacement dataset or model
was built. Publication and repository maintenance do not reopen model selection
or the final holdout.

## Publication detour and consolidation

The first public release on 2026-10-14 used a source-only export and a separate
local repository. It had two publication commits (`bb02f33`, `686c820`) rather
than this development history. The exported code passed hosted CI, but the
arrangement obscured the progression of the work and split the working workflow.
Those two commits are publication provenance, not original development commits.

The subsequent decision is to use the original repository and its full history,
name the project **Crude Oil Direction**, and remove the extra working copy.
There is no new sanitized history or replacement set of development hashes.
Older report excerpts remain in their original commits. Their earlier omission
from current reports was a precaution, not a determination that those particular
excerpts were prohibited. Raw source captures, private backups, full prediction
artifacts and environments remain ignored and local.

See [publication chronology](public-release.md) and its linked execution report
for completed publication steps and CI evidence. This history preserves the
experiments, unsuccessful approaches and corrective decisions as part of the
same project.
