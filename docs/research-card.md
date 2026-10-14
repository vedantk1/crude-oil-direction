# Crude Oil Direction: research card

## Question and intended use

Can a small, free-data model predict whether WTI spot is higher five fixed US
business days after the forecast date? Crude Oil Direction is a forecasting
software project and reproducible research record. It predicts a direction
probability, not a price level. WTI spot is a benchmark rather than a directly tradable security;
there is no transaction-cost, execution or futures-roll backtest.

## Evidence and design

ALFRED WTI/Brent vintages support a qualified replay from April 2011. EIA
inventories use 2011–2022 archived reports, with known gaps and a quarantined
release. Archives record dated history but cannot prove exact publication
timestamps, every initial release or historical file immutability. Keep these
limits separate from the software's causal date gates.

Initial training is 2011–2016, followed by expanding chronological refits in
2017–2022. There are 310 development forecasts, 301 scored outcomes and nine
unscored states. Five-business-day targets can overlap; observations are not
independent trials. Availability filtering determines the active label state
at each fit; it preserves revisions, withdrawals and explicit missing values.
Each training example's inputs are frozen at its original issuance.

The 2023-onward final holdout remains closed. The public workflow builds only
pre-2023 numerical history; full archive bytes may be hashed for integrity.
Original retained datasets/results remain preserved under ignored local data.

## What the experiments established

- WTI-only logistic regression did not beat the simple baselines.
- Brent added five correct direction predictions but worsened probability
  quality. This was a small, mixed result.
- Inventories added nine correct predictions against WTI-only and slightly
  improved probabilities; availability and missingness still limit interpretation.
- C=0.1 inventory regularization added two correct predictions versus C=1,
  with better Brier/log-loss scores in each development year. It still did not
  beat constant 0.5 or historical prevalence on probability quality.
- Four-year rolling training lost seven correct predictions against expanding
  C=0.1 and worsened aggregate probability quality.
- Small boosted trees lost nine correct predictions against expanding C=0.1.
  Some aggregate probability scores improved, but yearly behavior was mixed
  and simple probabilities remained better.
- The one additional business-day availability check tests timing robustness
  with unchanged forecasts and targets. It scores 53.82% with slightly better aggregate probabilities, but changes
  106 of 301 directions and has mixed yearly differences. Its result is recorded
  in [evaluation](evaluation.md); it is not used to select a new winner.

Original-clock expanding C=0.1 has the highest direction accuracy among the
model-search experiments at 53.16%; always-up achieves 55.81% because outcomes skew upward.
Accuracy alone does not establish useful skill. Repeated retrospective choices
on the same development years further limit confirmatory claims. No significance
or profitable-strategy claim is made.

## Frozen-phase policy

No more model/feature/lag/threshold search belongs to this phase. Preserve its
negative results and protocol history. Future research should begin with a
documented hypothesis, data-availability justification, evaluation plan and
decision criteria before collecting model results. A larger AI model is not
justified by the roughly weekly, small effective sample used here.

Any final holdout evaluation needs a separately agreed final model/protocol and
explicit authorization. Publishing code requires no claim of a successful
predictor: the project's value is reproducible evidence, causal evaluation and
an honest account of unsuccessful approaches. Current results do not justify
using the model for price forecasts, trading decisions or financial deployment.

## Reproducibility and distribution

Fresh workflows create their own references and artifact IDs. Synthetic tests
and CI exercise the pipeline without market downloads; they cannot establish
live-provider behavior or model skill. New source captures may differ from
retained evidence. See [reproducibility](reproducibility.md),
[contributor guidance](../CONTRIBUTING.md) and [data rights](../DATA_LICENSE.md).
Original code/docs/synthetic fixtures use MIT; third-party data do not inherit it.


## Current distribution and data-use status

The public demonstration is synthetic-only. The current published FRED terms
include a training-use restriction covering ALFRED; that use is unresolved,
independently of the permission to publish original MIT code. The source-price
reports now omit quotations while local backup files remain byte-exact. The
original Git history retains older report excerpts unchanged; their earlier
omission was a publication precaution, not an established prohibition on those
particular excerpts. The project uses one original repository with private
captures and artifacts excluded by Git ignore rules. See the
[publication workflow](public-release.md) and [development history](development-history.md).

The [free-source audit](free-training-data-audit.md) identifies licensed monthly
World Bank crude prices and EIA's own survey aggregates for a separate future
phase. Neither is a verified daily-WTI/vintage replacement. Existing findings,
model selection and the final holdout are unchanged. No new real-data model
was fitted during release preparation.
