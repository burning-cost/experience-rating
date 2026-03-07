# Databricks notebook source
# MAGIC %md
# MAGIC # experience-rating: Full workflow demo
# MAGIC
# MAGIC This notebook demonstrates the complete API of the `experience-rating` library
# MAGIC on synthetic data representative of a UK motor and commercial lines book.
# MAGIC
# MAGIC Sections:
# MAGIC 1. Installation
# MAGIC 2. UK Standard NCD Scale
# MAGIC 3. Stationary distribution and expected premium factors
# MAGIC 4. Optimal claiming thresholds
# MAGIC 5. Experience modification factors
# MAGIC 6. Schedule rating
# MAGIC 7. Running the test suite

# COMMAND ----------

# MAGIC %pip install experience-rating polars scipy numpy pytest

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. UK Standard NCD Scale

# COMMAND ----------

from experience_rating import BonusMalusScale, BonusMalusSimulator, ClaimThreshold

scale = BonusMalusScale.from_uk_standard()
print(f"Scale has {len(scale)} levels\n")
print(scale.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition matrix at 10% claim frequency

# COMMAND ----------

import numpy as np

T = scale.transition_matrix(claim_frequency=0.10)
print("Transition matrix (rows = from level, cols = to level):")
print(np.round(T, 3))
print(f"\nRow sums (should all be 1.0): {T.sum(axis=1).round(6)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Stationary Distribution

# COMMAND ----------

sim = BonusMalusSimulator(scale, claim_frequency=0.10, rng_seed=42)

# Analytical (eigenvector)
dist_a = sim.stationary_distribution(method="analytical")
print("Analytical stationary distribution:")
print(dist_a)

# COMMAND ----------

# Simulation (50k policyholders, 100 years)
dist_s = sim.stationary_distribution(method="simulation")
print("Simulation-based stationary distribution:")
print(dist_s)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expected premium factor at steady state

# COMMAND ----------

for freq in [0.05, 0.10, 0.15, 0.20, 0.30]:
    s = BonusMalusSimulator(scale, claim_frequency=freq)
    epf = s.expected_premium_factor()
    print(f"Claim frequency {freq:.0%} -> Expected premium factor: {epf:.4f} ({(1-epf)*100:.1f}% avg NCD)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Policyholder simulation over time

# COMMAND ----------

sim2 = BonusMalusSimulator(scale, claim_frequency=0.10, rng_seed=0)
df_sim = sim2.simulate(n_policyholders=50_000, n_years=20)
print(f"Simulation output shape: {df_sim.shape}")
print(df_sim.head(20))

# COMMAND ----------

# Check convergence: year 20 distribution vs analytical stationary
import polars as pl

year20 = (
    df_sim.filter(pl.col("year") == 20)
    .select(["level", "proportion"])
    .rename({"proportion": "simulated_year20"})
)
dist_a_clean = dist_a.select(["level", "stationary_prob"])

comparison = year20.join(dist_a_clean, on="level").with_columns(
    (pl.col("simulated_year20") - pl.col("stationary_prob")).abs().alias("abs_diff")
)
print("Year 20 simulation vs analytical stationary distribution:")
print(comparison)
print(f"\nMax absolute difference: {comparison['abs_diff'].max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Optimal Claiming Thresholds

# COMMAND ----------

ct = ClaimThreshold(scale, discount_rate=0.05)

# Full analysis at £800/year base premium
analysis = ct.full_analysis(annual_premium=800.0, years_horizon=3)
print("Claiming thresholds (3-year horizon, £800 annual premium):")
print(analysis)

# COMMAND ----------

# Threshold curves for a high-NCD customer (level 9, 65% NCD, £280/year after discount)
level9_premium = 800.0 * 0.35  # after 65% NCD
curve = ct.threshold_curve(current_level=9, annual_premium=level9_premium, max_horizon=8)
print(f"\nThreshold curve for 65% NCD customer (actual premium £{level9_premium:.0f}/yr):")
print(curve)

# COMMAND ----------

# Demonstrate optimal claiming decisions
test_cases = [
    (9, 150, "small scratch"),
    (9, 600, "moderate bumper"),
    (9, 2500, "significant damage"),
    (5, 300, "mid-NCD small claim"),
    (0, 500, "no NCD"),
]
print(f"{'Level':>6} {'Amount':>8} {'Description':>25} {'Should Claim?':>15}")
print("-" * 60)
for level, amount, desc in test_cases:
    prem = 800 * scale.levels[level].premium_factor
    should = ct.should_claim(level, amount, prem, years_horizon=3)
    print(f"{level:>6} {amount:>7}  {desc:>25} {'YES' if should else 'NO':>15}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Experience Modification Factors

# COMMAND ----------

from experience_rating import ExperienceModFactor
from experience_rating.experience_mod import CredibilityParams

# Standard parameters for a mid-size commercial account
params = CredibilityParams(credibility_weight=0.65, ballast=8_000.0)
emod = ExperienceModFactor(params)

# Synthetic commercial book
import numpy as np
rng = np.random.default_rng(99)

n_risks = 200
expected = rng.uniform(5_000, 100_000, n_risks)
# Actual losses: mostly near expected, some outliers
actual = expected * rng.lognormal(mean=0.0, sigma=0.5, size=n_risks)

portfolio = pl.DataFrame({
    "risk_id": [f"COM{i:04d}" for i in range(n_risks)],
    "sector": rng.choice(["Manufacturing", "Retail", "Office", "Warehouse"], n_risks).tolist(),
    "expected_losses": expected.tolist(),
    "actual_losses": actual.tolist(),
})

result = emod.predict_batch(portfolio, cap=2.5, floor=0.4)
print(f"Portfolio of {n_risks} commercial risks:")
print(result.head(10))

# COMMAND ----------

# Distribution of mod factors
mod_stats = result["mod_factor"].describe()
print("\nMod factor distribution:")
print(mod_stats)

debits = (result["mod_factor"] > 1.0).sum()
credits = (result["mod_factor"] < 1.0).sum()
neutral = (result["mod_factor"] == 1.0).sum()
print(f"\nDebits (mod > 1.0): {debits}")
print(f"Credits (mod < 1.0): {credits}")
print(f"Neutral (mod = 1.0): {neutral}")

# COMMAND ----------

# Sensitivity analysis: how does the mod respond to different loss outcomes?
sensitivity = emod.sensitivity(expected_losses=30_000, n_points=40)
print("\nSensitivity of mod factor to actual losses (expected = £30,000):")
print(sensitivity)

# COMMAND ----------

# Exposure-based credibility
emod_sq = ExperienceModFactor.from_exposure(
    actual_exposure=400,
    full_credibility_exposure=1000,
    ballast=5_000.0,
    credibility_formula="square_root",
)
print(f"\nExposure-based credibility (square root, 400/1000 years):")
print(f"A = {emod_sq.params.credibility_weight:.4f}")
print(f"Mod for £20k expected, £28k actual: {emod_sq.predict(20_000, 28_000):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Schedule Rating

# COMMAND ----------

from experience_rating import ScheduleRating

sr = ScheduleRating(max_total_debit=0.25, max_total_credit=0.25)
sr.add_factor("Premises", -0.10, 0.10, "Physical condition and maintenance of premises")
sr.add_factor("Management", -0.07, 0.07, "Management quality, experience, and track record")
sr.add_factor("Risk_Controls", -0.08, 0.08, "Adequacy of risk management and controls")

print("Schedule rating factors:")
print(sr.summary())

# COMMAND ----------

# Rate individual risks
examples = [
    {"Premises": 0.05, "Management": 0.03, "Risk_Controls": -0.02},
    {"Premises": -0.08, "Management": -0.05, "Risk_Controls": -0.06},
    {"Premises": 0.0, "Management": 0.0, "Risk_Controls": 0.0},
    {"Premises": 0.10, "Management": 0.07, "Risk_Controls": 0.08},  # capped
]

print("Schedule rating results:")
for features in examples:
    factor = sr.rate(features)
    print(f"  {features} -> Schedule factor: {factor:.4f}")

# COMMAND ----------

# Batch schedule rating
portfolio_sched = pl.DataFrame({
    "risk_id": [f"COM{i:04d}" for i in range(5)],
    "Premises": [0.05, -0.08, 0.00, 0.10, -0.05],
    "Management": [0.03, -0.05, 0.00, 0.07, 0.02],
    "Risk_Controls": [-0.02, -0.06, 0.00, 0.08, -0.03],
})

sched_result = sr.rate_batch(portfolio_sched)
print("\nBatch schedule rating:")
print(sched_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run the test suite

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /tmp && pip install experience-rating pytest -q && python -m pytest --pyargs experience_rating -v 2>&1 | tail -40

# COMMAND ----------

# Alternative: run tests from repo if mounted
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "/path/to/experience-rating/tests/", "-v", "--tb=short"],
    capture_output=True,
    text=True,
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print(result.stderr[-1000:])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `experience-rating` library provides:
# MAGIC
# MAGIC **BonusMalusScale** — define any NCD/BM system as a list of levels and transition
# MAGIC rules. `from_uk_standard()` gives the ABI 10-level NCD scale out of the box.
# MAGIC
# MAGIC **BonusMalusSimulator** — Monte Carlo simulation of policyholder flows plus
# MAGIC analytical stationary distribution via eigenvector decomposition. The two methods
# MAGIC should agree; when they don't, that's a signal your BM design is non-ergodic.
# MAGIC
# MAGIC **ClaimThreshold** — NPV analysis of whether a policyholder should absorb a
# MAGIC small loss to protect their NCD. The threshold at 65% NCD with a 3-year
# MAGIC horizon and £280 premium is typically in the £300-600 range, which matches
# MAGIC what experienced brokers tell their clients.
# MAGIC
# MAGIC **ExperienceModFactor** — NCCI-style credibility-weighted experience modification.
# MAGIC The ballast parameter is the key design choice: set it to limit the swing from
# MAGIC a single large loss.
# MAGIC
# MAGIC **ScheduleRating** — validated debit/credit schedule with aggregate caps. The
# MAGIC validation at entry time is deliberate — silent out-of-range factors are a
# MAGIC compliance risk.
