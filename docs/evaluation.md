# Chronological development evaluation

This protocol compares two simple rules with a WTI-only logistic regression. It is fixed before computing the first model-comparison results. The underlying price target and availability convention remain those in the [dataset specification](dataset.md).

## Time boundaries and final holdout

| Period | Role |
| --- | --- |
| 2011-04-07 through 2016-12-31 | Initial training history |
| 2017-01-01 through 2022-12-31 | Expanding chronological development replay |
| 2023-01-01 onward | Closed final holdout |

Each development forecast refits using strictly earlier forecast rows whose labels were available at that issuance cutoff. Earlier development outcomes can join later training only once available. There is no random split. The forecast and its exact target date must both precede the holdout boundary; crossing targets are excluded. Scoring uses first-available outcomes known by 2022-12-31. A prediction whose outcome is still unavailable remains explicitly unscored.

The final holdout is excluded from fitting, prediction, scoring, and tuning. The loader checks the full source-file hashes but excludes holdout rows before materializing their feature or label values. Earlier dataset auditing reported full-history coverage, so this is a holdout from model development, not a claim that the underlying data have never been read. There is no CLI switch to run it.

## Fixed models

| Model | Prediction rule | Probability estimate |
| --- | --- | --- |
| Majority | Most frequent class in that issuance's eligible training labels; ties choose up | Training up-class frequency |
| Momentum | Up if the available five-business-day WTI dollar change is positive; zero means not up | None |
| WTI logistic regression | Six declared WTI inputs, regularized logistic regression; probability at least 0.5 means up | Fitted up-class probability |

Missing momentum falls back to that fit's majority prediction and is flagged. It is not assigned an invented probability. Logistic regression uses median imputation and standard scaling fitted only on that training window; an entirely missing training column remains present and is filled with zero. No missingness indicators or Brent inputs are added. Parameters are fixed at L2 regularization, `C=1`, `lbfgs`, `max_iter=2000`, no class weights, and `random_state=0`. No parameter or threshold search is performed.

Each fit requires at least 150 eligible training rows and both classes. Insufficient history, invalid inputs, or nonconvergence fail the evaluation rather than silently substituting a different classifier. These rules are recorded in `oil_eval/protocol.py`.

The preprocessing pipeline follows scikit-learn's guidance to fit transforms on training data and apply them to later observations. References: [leakage prevention](https://scikit-learn.org/1.7/common_pitfalls.html), [logistic regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html), [imputer behavior](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html).

## Scoring and interpretation

All models are compared on the same scored development forecast dates. Report accuracy, balanced accuracy, confusion counts, and yearly breakdowns. For majority and logistic probabilities also report ROC AUC, Brier score, and natural-log loss; lower Brier/log loss is better. Log loss clips probabilities to `[1e-15, 1-1e-15]`. Balanced accuracy and AUC are undefined when truth contains only one class. Momentum has no probability metrics.

Training uses active historical label versions, including corrections and withdrawals; scoring intentionally freezes the first jointly available outcome. Later revisions or withdrawals cannot alter that issuance's earlier training data. Events and scoring labels from 2023 onward are excluded, even for older forecast IDs. Source version end dates crossing the holdout boundary are masked in the development view.

The primary comparison is paired development accuracy, with probability quality and yearly stability reported alongside it. A few hundred weekly examples and potentially overlapping horizons do not justify independent-sample significance claims. Development performance is not final holdout performance, trading profitability, or a deployment claim.

## Checkpoints

The first evaluation checkpoint locks this protocol and tests the data boundary. The next adds the runner, models, tests, and measured development results. The final holdout remains closed at both checkpoints. Results and reproducible commands will be added here after the development run.
