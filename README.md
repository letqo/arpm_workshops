# ARPM Workshop Notebooks

Exercises from the ARPM (Advanced Risk and Portfolio Management) bootcamp, covering
mean-covariance statistics, probabilistic machine learning, and (in progress) time
series / reinforcement learning applied to markets and credit risk.

## Day 1 — Linear Mean-Covariance Statistics (PCA on equity returns)

Builds a full factor-analysis pipeline on daily returns for 14 large-cap stocks
(`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `TSLA`, `AMD`, `INTC`, `CRM`, `ORCL`,
`ADBE`, `CSCO`, `IBM`):

- **Flexible probabilities** — exponentially decaying weights (60-day half-life)
  instead of an equal-weighted covariance, plus the effective number of scenarios
  (ENS) to quantify how much of the sample the weighting actually uses.
- **PCA via eigendecomposition** — the weighted covariance is rescaled to a
  correlation matrix first, so PCA isn't dominated by whichever stock happens to be
  most volatile.
  - PC1 explains **~38%** of total co-movement variance (eigenvalue 5.35 / 14).
  - PC1 loads positively on every stock (no sign flips), with `MSFT` (0.355),
    `AMZN` (0.350) and `META` (0.291) carrying the largest weights — read as a broad
    tech/market factor.
- **Out-of-sample validation** — PC1/PC2 are fit on 2024 data only, then used to
  reconstruct 2025 returns. Actual-vs-reconstructed correlation per stock ranges from
  0.55 (`INTC`) to 0.83 (`AMZN`), i.e. the factor structure found in-sample holds up
  reasonably well out-of-sample, especially for the names most exposed to it.
- **Factor portfolios** — eigenvectors are converted into implementable portfolio
  weights and their cumulative performance is plotted; the PC1 (broad tech) portfolio
  is choppy through 2024 before a sharp drawdown (to almost -0.9) around March–May
  2025, then a partial recovery.

## Day 2 — Probabilistic Machine Learning (credit risk model)

An end-to-end credit-scoring pipeline on the German Credit dataset (OpenML `data_id=31`):

- **Preprocessing** — one-hot encoding for categorical features, `QuantileTransformer`
  (normal output) for continuous features, train/test split.
- **Model** — logistic regression with L1 (lasso) regularization (`lambda = 3`).
- **In-sample performance** — ROC/AUC ≈ **0.82**.
- **Cutoff selection** — compares the statistically optimal cutoff (Youden's J) to a
  deliberately conservative **business cutoff of 0.72**, reasoning explicitly about
  asymmetric costs (approving a bad-credit applicant is worse for the bank than
  rejecting a good one). At that cutoff: false positive rate ≈ **5.2%**, false
  negative rate ≈ **22.7%**.
- **Out-of-sample performance** — test AUC ≈ **0.78** (a ~0.04 drop from train),
  interpreted as reasonable generalization rather than overfitting.
- **Application** — the fitted pipeline is applied to a new applicant's raw data to
  produce a probability of good credit and a lending decision.

## Day 3 — Time Series and Reinforcement Learning

*In progress — not yet included in this repo.*

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` (`OneHotEncoder`, `QuantileTransformer`,
logistic regression, ROC/AUC) · `matplotlib`
