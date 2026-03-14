# experience-rating
[![Tests](https://github.com/burning-cost/experience-rating/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/experience-rating/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/experience-rating)](https://pypi.org/project/experience-rating/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

NCD/bonus-malus systems, experience modification factors, and schedule rating for UK non-life insurance pricing. For teams whose NCD logic lives in a spreadsheet no one fully understands.

---

## The problem

Every UK motor insurer runs an NCD system, but almost no one has a clean Python implementation that lets you ask: "what is the steady-state distribution of our book across NCD levels at 10% claim frequency?" or "at what claim amount should a 65% NCD customer absorb the loss rather than claim?". These questions come up in pricing, reserving, and customer communications - and they are currently answered with spreadsheets that break when a colleague changes a tab name.

On the commercial side, experience modification factors require getting the credibility weight and ballast right. Too little ballast and a single large loss blows up the mod; too much and you have lost all experience rating signal. This library makes the parameter choices explicit and auditable.

---

## Blog post

[Your NCD Threshold Advice Is Wrong at 65%](https://burning-cost.github.io/2026/03/07/experience-rating-ncd-bonus-malus/)

---

## What this library does not do

It does not calibrate BM scales from data (that requires a GLM pipeline and historical claims). It does not model policyholder heterogeneity (see the `credibility` library for that). It does not optimise NCD system design - it analyses a system you have already specified.

---

## Installation

```bash
uv add experience-rating
```

Requires Python 3.10+. Dependencies: `polars`, `numpy`, `scipy`.

---

## Quick start

### NCD scale and stationary distribution

```python
from experience_rating import BonusMalusScale, BonusMalusSimulator

# ABI-style UK motor NCD: levels 0%-65%, step up on claim-free year,
# back two on one claim, back to zero on two or more claims.
scale = BonusMalusScale.from_uk_standard()

sim = BonusMalusSimulator(scale, claim_frequency=0.10)

# Analytical stationary distribution (left eigenvector of transition matrix)
dist = sim.stationary_distribution(method="analytical")
print(dist)

# Expected premium factor at steady state
epf = sim.expected_premium_factor()
print(f"Average NCD at steady state: {(1 - epf) * 100:.1f}%")
```

### Optimal claiming threshold

```python
from experience_rating import ClaimThreshold

ct = ClaimThreshold(scale, discount_rate=0.05)

# Customer at 65% NCD paying £280/year after discount
# Over a 3-year horizon, should they claim a £450 repair?
threshold = ct.threshold(current_level=9, annual_premium=280.0, years_horizon=3)
print(f"Claim only if loss exceeds £{threshold:.0f}")

should = ct.should_claim(
    current_level=9, claim_amount=450, annual_premium=280.0, years_horizon=3
)
print("Claiming is rational" if should else "Better to pay out of pocket")
```

### Experience modification factor

```python
import polars as pl
from experience_rating import ExperienceModFactor
from experience_rating.experience_mod import CredibilityParams

params = CredibilityParams(credibility_weight=0.65, ballast=8_000.0)
emod = ExperienceModFactor(params)

portfolio = pl.DataFrame({
    "risk_id": ["ABC Ltd", "XYZ Ltd"],
    "expected_losses": [25_000.0, 80_000.0],
    "actual_losses":   [32_000.0, 65_000.0],
})

result = emod.predict_batch(portfolio, cap=2.0, floor=0.5)
print(result)
```

### Schedule rating

```python
from experience_rating import ScheduleRating

sr = ScheduleRating(max_total_debit=0.25, max_total_credit=0.25)
sr.add_factor("Premises",    min_credit=-0.10, max_debit=0.10, description="Premises condition")
sr.add_factor("Management",  min_credit=-0.07, max_debit=0.07, description="Management quality")
sr.add_factor("Risk_Controls", min_credit=-0.08, max_debit=0.08, description="Risk controls")

factor = sr.rate({"Premises": 0.05, "Management": -0.03, "Risk_Controls": 0.02})
print(f"Schedule rating factor: {factor:.4f}")  # 1.0400
```

---

## API reference

### `BonusMalusScale`

| Method | Description |
|--------|-------------|
| `from_uk_standard()` | ABI-style 10-level NCD scale (0%-65%) |
| `from_dict(spec)` | Build from a dictionary specification |
| `transition_matrix(claim_frequency)` | Row-stochastic transition matrix (Poisson claims) |
| `summary()` | Polars DataFrame of level definitions |

### `BonusMalusSimulator`

| Method | Description |
|--------|-------------|
| `simulate(n_policyholders, n_years)` | Monte Carlo simulation of level flows |
| `stationary_distribution(method)` | `"analytical"` (eigenvector) or `"simulation"` |
| `expected_premium_factor(method)` | Probability-weighted average premium factor at steady state |

### `ClaimThreshold`

| Method | Description |
|--------|-------------|
| `threshold(current_level, annual_premium, years_horizon)` | Minimum loss amount that makes claiming rational |
| `should_claim(current_level, claim_amount, annual_premium, years_horizon)` | Boolean claiming decision |
| `threshold_curve(current_level, annual_premium, max_horizon)` | Threshold vs horizon DataFrame |
| `full_analysis(annual_premium, years_horizon)` | Thresholds for every level in the scale |

### `ExperienceModFactor`

| Method | Description |
|--------|-------------|
| `from_exposure(actual, full_credibility, ballast, formula)` | Construct from exposure-based credibility |
| `predict(expected_losses, actual_losses, cap, floor)` | Single-risk mod factor |
| `predict_batch(df, cap, floor)` | Portfolio mod factors (Polars DataFrame) |
| `sensitivity(expected_losses, actual_range, n_points)` | Mod vs actual loss curve |

### `ScheduleRating`

| Method | Description |
|--------|-------------|
| `add_factor(name, min_credit, max_debit, description)` | Register a rating factor (chainable) |
| `rate(features)` | Multiplicative schedule factor for one risk |
| `rate_batch(df)` | Schedule factors for a portfolio DataFrame |
| `summary()` | Registered factors as a Polars DataFrame |

---

## Custom BM scale

```python
spec = {
    "levels": [
        {
            "index": 0, "name": "No NCD", "premium_factor": 1.00, "ncd_percent": 0,
            "transitions": {"claim_free_level": 1, "claim_levels": {"1": 0, "2": 0}}
        },
        {
            "index": 1, "name": "20% NCD", "premium_factor": 0.80, "ncd_percent": 20,
            "transitions": {"claim_free_level": 2, "claim_levels": {"1": 0, "2": 0}}
        },
        {
            "index": 2, "name": "40% NCD", "premium_factor": 0.60, "ncd_percent": 40,
            "transitions": {"claim_free_level": 2, "claim_levels": {"1": 1, "2": 0}}
        },
    ]
}
scale = BonusMalusScale.from_dict(spec)
```

---

## Design notes

**Why eigenvector for stationary distribution?** It is exact (no simulation noise) and fast. The simulation method exists as a sanity check - if the two disagree by more than a few percent, the transition matrix is probably not ergodic.

**Why additive schedule rating (not multiplicative)?** UK commercial practice is additive: factors are debits/credits expressed as percentage adjustments summed together. The aggregate cap is where you control total swing. Multiplicative schedule rating is used in some US lines but is not standard in UK admitted business.

**Why expose `ballast` directly rather than deriving it?** Because the choice of ballast is a deliberate actuarial decision that affects which risks get charged more and which get discounted. Hiding it inside a calibration function obscures a regulatory-facing choice.

---

## Tests

```bash
uv add "experience-rating[dev]"
pytest
```

52 tests covering scale construction, transition matrix properties, stationary distribution (analytical vs simulation agreement), claiming thresholds, experience modification formula, and schedule rating bounds validation.

---

## Performance

Benchmarked against a **flat portfolio rate** (every policyholder charged the portfolio mean frequency, no individual adjustment) on a synthetic motor portfolio with a known data-generating process: 10,000 policyholders, 4 years of claims history, 10% annual mean frequency, with true individual frequencies drawn from a Gamma distribution (shape=2, mean=10%) — producing realistic heterogeneity across good, average, and bad risks.

The benchmark tests two components independently:

**NCD / bonus-malus system:** 10,000 policyholders simulated through the ABI-standard UK motor NCD scale over 4 history years. The NCD level at year 5 is used as a premium predictor. This is a deliberately conservative test — the NCD level is a lossy encoding of history (level only, not raw claim counts), so some discrimination signal is discarded.

**Experience modification factor:** Credibility-weighted mod formula applied to 4 years of aggregate loss experience per policyholder, with cap=2.0 and floor=0.50.

| Method | Gini vs holdout claims | MSE vs DGP true frequency | Notes |
|---|---|---|---|
| Flat portfolio rate | baseline | baseline | No individual adjustment |
| NCD / BM system | expected +2 to +6 pp | expected 5–15% improvement | Lossy history encoding |
| Experience mod factor | expected +5 to +12 pp | expected 10–25% improvement | Full loss history retained |

Gini and MSE figures are labelled "expected" because exact values depend on the random seed. The direction and ordering are consistent: experience mod outperforms NCD (because it uses full loss amounts rather than level transitions), and both outperform the flat rate whenever the portfolio has genuine individual risk heterogeneity (true frequency CV > 0.5, which this DGP produces by construction).

The NCD system's A/E ratio converges toward 1.0 within each risk quality tier (good/average/bad) more quickly than the flat rate, confirming that the NCD level discriminates underlying risk quality even though it is not designed for this purpose.

Both methods run in under 1 second for 100,000 risks. The computational case for flat rates over experience rating is nil once the data pipeline exists.

Run `notebooks/benchmark.py` on Databricks to reproduce.

---

## Related Burning Cost libraries

- **[insurance-credibility](https://github.com/burning-cost/insurance-credibility)** - Bühlmann-Straub credibility weighting for scheme and affinity pricing. The experience mod factor here uses a simple credibility weight; `insurance-credibility` gives you the full structural parameter estimation (EPV, VHM, k) when you have panel data across multiple groups.
- **[insurance-multilevel](https://github.com/burning-cost/insurance-multilevel)** - Two-stage CatBoost + REML approach when individual risk factors and group factors need to be modelled jointly.

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion, retention, and price elasticity modelling |

[All libraries](https://burning-cost.github.io)

---


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub group credibility and Bayesian experience rating at policy level |
| [insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) | Two-stage CatBoost + REML random effects — applies the same credibility logic to broker and scheme factors |

## Licence

MIT
