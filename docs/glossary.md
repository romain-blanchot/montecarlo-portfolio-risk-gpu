# Glossary

Financial and technical terms used throughout this project.

---

## Risk Metrics

### Value at Risk (VaR)

The **α-quantile** of the loss distribution. VaR at confidence level α (e.g. 95%) answers:

> "With probability α, we will not lose more than VaR over the given time horizon."

$$\text{VaR}_\alpha = \inf\{l : P(\text{Loss} > l) \leq 1 - \alpha\}$$

This engine computes VaR as `np.percentile(losses, α × 100)`.

!!! warning "Convention"
    This engine uses the **loss-positive convention**: a positive VaR means the portfolio lost money. A negative VaR means the portfolio gained money even in the worst (1−α) fraction of scenarios.

### Expected Shortfall (ES)

Also called **Conditional VaR (CVaR)** or **Tail Loss**. The average loss in the worst (1−α) fraction of outcomes:

$$\text{ES}_\alpha = \mathbb{E}[\text{Loss} \mid \text{Loss} \geq \text{VaR}_\alpha]$$

ES is always ≥ VaR. It is considered a more robust risk measure because it captures the **severity** of tail losses, not just where the tail begins. ES is a **coherent risk measure** (subadditive), unlike VaR.

### Loss-Positive Convention

In this engine, **loss = −return**. A positive value means the portfolio declined; a negative value means it gained. All VaR and ES values follow this convention.

---

## Stochastic Model

### Geometric Brownian Motion (GBM)

The baseline price dynamics model. Each asset price $S_t$ follows:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

where:

- $\mu$ — annualized drift (expected return)
- $\sigma$ — annualized volatility
- $W_t$ — Wiener process (Brownian motion)

The closed-form terminal price over horizon $T$ is:

$$S_T = S_0 \exp\left[\left(\mu - \tfrac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right]$$

where $Z \sim \mathcal{N}(0, 1)$.

### Multivariate GBM

Extension to multiple correlated assets. Independent standard normals $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ are correlated via Cholesky decomposition:

$$\mathbf{Z}_{\text{corr}} = L \cdot \mathbf{Z}$$

where $L$ is the lower-triangular Cholesky factor of the covariance matrix $\Sigma = L L^\top$.

### Drift Vector

A vector $\boldsymbol{\mu} = (\mu_1, \ldots, \mu_n)$ of annualized expected returns, one per asset. Estimated from historical log returns.

### Covariance Matrix

A symmetric positive-definite matrix $\Sigma$ capturing both individual asset volatilities (diagonal) and pairwise co-movements (off-diagonal). Estimated from historical log returns and annualized.

### Cholesky Decomposition

Factorization of the covariance matrix $\Sigma = L L^\top$ into a lower-triangular matrix $L$. Used to generate correlated random samples from independent ones. The engine provides a pure-Python implementation in the domain layer and a GPU implementation via CuPy.

---

## Simulation

### Monte Carlo Simulation

A numerical method that estimates quantities by random sampling. This engine:

1. Generates $N$ independent standard normal random vectors
2. Correlates them via Cholesky decomposition
3. Computes terminal asset prices using the GBM formula
4. Derives portfolio-level returns and losses
5. Estimates VaR and ES from the empirical loss distribution

### Terminal Price

The simulated asset price $S_T$ at the end of the time horizon. Each simulation path produces one terminal price per asset.

### Time Horizon

The forward-looking period over which risk is measured, expressed in **trading days**. Common values: 1 day (regulatory), 10 days (Basel), 21 days (~1 month), 252 days (~1 year).

### Annualization Factor

A multiplier to convert per-period statistics to annual values. Automatically estimated from the median gap between historical dates:

| Median gap | Frequency | Factor |
|------------|-----------|--------|
| ≤ 5 days   | Daily     | 252    |
| ≤ 10 days  | Weekly    | 52     |
| ≤ 40 days  | Monthly   | 12     |
| ≤ 100 days | Quarterly | 4      |
| > 100 days | Annual    | 1      |

---

## Portfolio

### Portfolio

A collection of **positions** (asset + weight pairs). Weights must sum to 1.0 (within tolerance 1e-8). No duplicate tickers allowed.

### Position

A single holding: an `Asset` paired with a `Weight` representing its allocation in the portfolio.

### Weight

A portfolio allocation fraction, constrained to $[0, 1]$.

### Ticker

A stock exchange symbol (e.g. `AAPL`, `MSFT`, `BRK.B`). Normalized to uppercase, validated against the pattern `[A-Z0-9.\-]{1,10}`.

### Log Returns

The natural logarithm of price ratios between consecutive observations:

$$r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)$$

Log returns are preferred over simple returns because they are additive over time and approximately normally distributed.
