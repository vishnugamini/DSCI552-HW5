# DSCI 552 - Homework 5

Vishnu Gamini · GitHub: vishnugamini

## Files

```text
data/
  acute_inflammations/
    diagnosis.data
    diagnosis.names
  communities_and_crime/
    communities.data
    communities.names
  results/
    ... rules, cross-validation scores, coefficients, predictions, and errors
notebook/
  Gamini_Vishnu_HW5.ipynb
requirements.txt
```

## Run locally

Use Python 3.12. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

Open `notebook/Gamini_Vishnu_HW5.ipynb`, restart the kernel, and run all cells. Saved outputs are included. Paths work from the repository root or the notebook directory, and execution recreates `data/results/` without downloading data. On macOS, an XGBoost import error mentioning `libomp` requires the OpenMP runtime (`brew install libomp`).

## Methods

- Acute Inflammations: label-powerset encoding of both diagnosis labels, a full Gini decision tree, IF-THEN rules, and cost-complexity pruning with five stratified folds. The one-standard-error choice retains six leaves. A four-leaf alternative meets an explicit 80% validation-accuracy floor; its loss of the both-positive diagnosis class is documented.
- Communities and Crime: preserve source order, use the first 1,495 rows for training and the remaining 499 for testing, and remove the five nonpredictive identifiers. Median imputation and any standardization are fitted inside each of five training folds. Correlations, coefficients of variation, and exploratory plots use training data.
- Compare OLS, standardized ridge, original-scale and standardized LASSO, standardized principal component regression, and L1-regularized XGBoost trees. Penalties and the PCR component count are selected by validation MSE. The notebook lists both LASSO variable sets and reports held-out MSE, RMSE, and MAE.
- XGBoost uses 500 depth-three trees, learning rate 0.05, and cross-validation over `reg_alpha`. Its L1 penalty applies to constant leaf weights, rather than separate linear models at internal nodes. Random seeds are 42.

## Data sources

Original source data and descriptions are preserved. Both datasets are distributed under CC BY 4.0.

1. Czerniak, J. (2003). *Acute Inflammations*. UCI Machine Learning Repository. https://doi.org/10.24432/C5V59S.
2. Redmond, M. (2002). *Communities and Crime*. UCI Machine Learning Repository. https://doi.org/10.24432/C53W3X.
