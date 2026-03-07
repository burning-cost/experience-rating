# experience-rating

NCD/bonus-malus systems, experience modification factors, and schedule rating for UK non-life insurance pricing.

## The problem

Every UK motor insurer runs an NCD system, but almost no one has a clean Python implementation that lets you ask: "what is the steady-state distribution of our book across NCD levels at 10% claim frequency?" or "at what claim amount should a 65% NCD customer absorb the loss rather than claim?". These questions come up in pricing, reserving, and customer communications — and they're currently answered with spreadsheets that break when a colleague changes a tab name.

On the commercial side, experience modification factors require getting the credibility weight and ballast right. Too little ballast and a single large loss blows up the mod; too much and you've lost all experience rating signal. This library makes the parameter choices explicit and auditable.

## What this library does not do

It does not calibrate BM scales from data (that requires a GLM pipeline and historical claims). It does not model policyholder heterogeneity (see the `credibility` library for that). It does not optimise NCD system design — it analyses a system you've already specified.

## Installation

```bash
uv add experience-rating
```

Requires Python 3.10+. Dependencies: `polars`, `numpy`, `scipy`.

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

## API reference

### `BonusMalusScale`

| Method | Description |
|--------|-------------|
| `from_uk_standard()` | ABI-style 10-level NCD scale (0%–65%) |
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

## Design notes

**Why eigenvector for stationary distribution?** It is exact (no simulation noise) and
fast. The simulation method exists as a sanity check — if the two disagree by more than
a few percent, the transition matrix is probably not ergodic.

**Why additive schedule rating (not multiplicative)?** UK commercial practice is
additive: factors are debits/credits expressed as percentage adjustments summed together.
The aggregate cap is where you control total swing. Multiplicative schedule rating is used
in some US lines but is not standard in UK admitted business.

**Why expose `ballast` directly rather than deriving it?** Because the choice of
ballast is a deliberate actuarial decision that affects which risks get charged more
and which get discounted. Hiding it inside a calibration function obscures a
regulatory-facing choice.

## Tests

```bash
uv add "experience-rating[dev]"
pytest
```

52 tests covering scale construction, transition matrix properties, stationary
distribution (analytical vs simulation agreement), claiming thresholds, experience
modification formula, and schedule rating bounds validation.

## Licence

MIT
