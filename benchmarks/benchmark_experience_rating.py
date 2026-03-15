# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: experience-rating NCD/EM-factor vs flat portfolio rate
# MAGIC
# MAGIC **Library:** `experience-rating` — NCD bonus-malus systems, experience
# MAGIC modification factors (e-mods), and schedule rating for UK non-life insurance.
# MAGIC
# MAGIC **Baseline:** flat portfolio rate — every risk charged the same rate regardless
# MAGIC of claims history. The correct baseline: experience rating only adds value when
# MAGIC history is credible enough to separate genuine risk quality from noise.
# MAGIC
# MAGIC **Dataset:** 1,500 commercial risks with known latent risk quality (theta_i),
# MAGIC 3 years of claims history, variable exposures. We test two methods:
# MAGIC - NCD/BM system (BonusMalusScale) — stepwise adjustment by claim count
# MAGIC - Experience mod (ExperienceModFactor) — credibility-weighted adjustment
# MAGIC
# MAGIC **Date:** 2026-03-15
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC The flat rate is what you charge if you have no history or don't trust it.
# MAGIC The NCD system separates good/bad risks but ignores exposure and uses
# MAGIC hardcoded thresholds. The e-mod formula blends actual and expected losses
# MAGIC with a credibility weight calibrated to the data. We measure which approach
# MAGIC best predicts future loss costs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install experience-rating polars numpy scipy matplotlib pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from experience_rating import (
    BonusMalusScale,
    BonusMalusSimulator,
    ExperienceModFactor,
    CredibilityParams,
)

warnings.filterwarnings("ignore")
print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Commercial Portfolio
# MAGIC
# MAGIC - 1,500 risks. Each has a latent risk quality theta_i ~ Gamma(alpha, alpha),
# MAGIC   so E[theta]=1, Var[theta]=1/alpha. We use alpha=3 (moderate heterogeneity).
# MAGIC - 3 years of observed claims: N_t ~ Poisson(theta_i * base_rate * exposure_t)
# MAGIC - 1 future year held out for evaluation.
# MAGIC - Base annual premium = £2,000. Base claim rate = 0.12 per vehicle-year.

# COMMAND ----------

rng = np.random.default_rng(42)
N_RISKS = 1_500
N_YEARS_HIST = 3
ALPHA_GAMMA = 3.0     # portfolio heterogeneity
BASE_RATE   = 0.12    # base claim frequency
BASE_PREM   = 2_000.0

# True latent risks
theta = rng.gamma(ALPHA_GAMMA, 1.0/ALPHA_GAMMA, N_RISKS)

# Historical exposures (variable — tests exposure weighting)
exposures = rng.uniform(0.5, 1.5, (N_RISKS, N_YEARS_HIST))

# Observed historical claims
hist_claims = rng.poisson(theta[:, None] * BASE_RATE * exposures)

# Future (holdout) year for evaluation
exp_future  = rng.uniform(0.5, 1.5, N_RISKS)
fut_claims  = rng.poisson(theta * BASE_RATE * exp_future)

# Summary statistics
print(f"Portfolio: {N_RISKS:,} risks, {N_YEARS_HIST} year history")
print(f"True theta: mean={theta.mean():.3f}, std={theta.std():.3f}, CV={theta.std()/theta.mean():.3f}")
print(f"Historical claims: {hist_claims.sum():,} total, {hist_claims.sum()/exposures.sum():.4f} per veh-yr")
print(f"Future claims:     {fut_claims.sum():,} total, {fut_claims.sum()/exp_future.sum():.4f} per veh-yr")
print()
print("Claims history distribution:")
for n, cnt in zip(*np.unique(hist_claims.sum(axis=1), return_counts=True)):
    if n > 8: break
    print(f"  {int(n)} total claims: {cnt} risks ({cnt/N_RISKS:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline 1: Flat Portfolio Rate

# COMMAND ----------

t0_base = time.perf_counter()
# Every risk gets the portfolio average rate
flat_factor = np.ones(N_RISKS)
flat_premium = flat_factor * BASE_PREM
base_time = time.perf_counter() - t0_base

# Predicted future claims under flat rate
flat_pred_claims = BASE_RATE * exp_future  # same prediction for everyone

print(f"Baseline time: {base_time:.5f}s")
print(f"Flat premium: £{flat_premium.mean():.0f} for all {N_RISKS:,} risks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NCD / Bonus-Malus System

# COMMAND ----------

# UK standard 10-level NCD scale from the library
scale = BonusMalusScale.from_uk_standard()

print("UK Standard NCD Scale:")
print(scale.summary().to_pandas().to_string(index=False))

# Assign each risk to a BM level based on 3-year history
# Rule: start at level 0. Apply transitions year by year.
def apply_bm_transitions(n_claims_history, scale, start_level=0):
    """Simulate BM level transitions over the claim history."""
    level = start_level
    for yr_claims in n_claims_history:
        n = int(yr_claims)
        level = scale.levels[level].transitions.next_level(n)
    return level

t0_ncd = time.perf_counter()
bm_levels = np.array([
    apply_bm_transitions(hist_claims[i], scale)
    for i in range(N_RISKS)
])
bm_factors = np.array([scale.levels[l].premium_factor for l in bm_levels])
bm_premiums = bm_factors * BASE_PREM
ncd_time = time.perf_counter() - t0_ncd

# NCD factor is applied as a premium relativities:
# predicted rate = base_rate * bm_factor (higher NCD = lower factor = safer)
bm_pred_rate = BASE_RATE * bm_factors

print(f"\nNCD scoring time: {ncd_time:.4f}s")
print(f"BM factor distribution:")
for lvl in range(10):
    cnt = (bm_levels == lvl).sum()
    print(f"  Level {lvl} ({scale.levels[lvl].name}): {cnt} risks ({cnt/N_RISKS:.1%}), factor={scale.levels[lvl].premium_factor:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Experience Modification Factor (credibility-weighted)

# COMMAND ----------

# Estimate portfolio parameters for credibility
# Within-risk variance: sigma^2 = E[N/exposure] for Poisson = BASE_RATE
# Between-risk variance: tau^2 = Var(theta) * BASE_RATE^2 * mean_exposure
# kappa = sigma^2 / (tau^2 * mean_exposure) = 1 / (Var(theta) * mean_exposure)

# Empirical moment estimator
obs_rates = hist_claims.sum(axis=1) / exposures.sum(axis=1)
portfolio_mean = float(obs_rates.mean() / BASE_RATE)  # ~1.0
tau2_empirical = max(float(obs_rates.var()) - float(BASE_RATE / exposures.mean()), 1e-6)
sigma2 = float(BASE_RATE / exposures.mean())
kappa_hat = sigma2 / tau2_empirical

credibility_at_3yr = 3.0 * exposures.mean() / (3.0 * exposures.mean() + kappa_hat)

print(f"Empirical kappa (k): {kappa_hat:.4f}")
print(f"Credibility at 3yr avg exposure: {credibility_at_3yr:.2%}")

# ExperienceModFactor uses formula: Mod = (A*actual + (1-A)*expected + B) / (expected + B)
# We derive A from exposure and kappa, set B = expected_losses * (1-A)
A_cred = np.array([
    float(exp_i / (exp_i + kappa_hat))
    for exp_i in exposures.sum(axis=1)
])
BALLAST = 5_000.0  # standard commercial ballast

t0_emod = time.perf_counter()
emod_factors = np.zeros(N_RISKS)
for i in range(N_RISKS):
    params = CredibilityParams(credibility_weight=float(A_cred[i]), ballast=BALLAST)
    emod = ExperienceModFactor(params)
    actual_losses = float(hist_claims[i].sum()) * BASE_PREM / BASE_RATE
    expected_losses = float(exposures[i].sum()) * BASE_RATE * BASE_PREM
    emod_factors[i] = emod.predict(expected_losses, actual_losses, cap=2.5, floor=0.5)
emod_time = time.perf_counter() - t0_emod

emod_premiums = emod_factors * BASE_PREM
emod_pred_rate = BASE_RATE * emod_factors

print(f"\nE-mod scoring time: {emod_time:.4f}s")
print(f"E-mod factor distribution:")
print(f"  mean={emod_factors.mean():.4f}, std={emod_factors.std():.4f}")
print(f"  p10={np.percentile(emod_factors,10):.4f}, p50={np.percentile(emod_factors,50):.4f}, p90={np.percentile(emod_factors,90):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Metrics

# COMMAND ----------

# Future predicted claims for each method
true_future_rate = theta * BASE_RATE  # oracle

def gini(y_true, y_pred, w=None):
    if w is None: w = np.ones_like(y_true)
    order = np.argsort(y_pred)
    ys, ws = y_true[order], w[order]
    cw = np.cumsum(ws)/ws.sum(); cy = np.cumsum(ys*ws)/(ys*ws).sum()
    return float(2*np.trapz(cy, cw) - 1)

def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def ae_mae_by_band(y_true, y_pred, n=5):
    cuts = pd.qcut(y_pred, n, labels=False, duplicates="drop")
    aes = [float(y_true[cuts==q].sum() / max(y_pred[cuts==q].sum(), 1e-10)) for q in range(n) if (cuts==q).sum() >= 5]
    return float(np.mean(np.abs(np.array(aes)-1.0)))

# Compare predicted future rate vs true theta-based rate
gini_flat  = gini(true_future_rate, flat_factor * BASE_RATE, w=exp_future)
gini_ncd   = gini(true_future_rate, bm_pred_rate,             w=exp_future)
gini_emod  = gini(true_future_rate, emod_pred_rate,           w=exp_future)

rmse_flat  = rmse(true_future_rate, flat_factor * BASE_RATE)
rmse_ncd   = rmse(true_future_rate, bm_pred_rate)
rmse_emod  = rmse(true_future_rate, emod_pred_rate)

ae_flat  = ae_mae_by_band(fut_claims/exp_future, flat_factor*BASE_RATE)
ae_ncd   = ae_mae_by_band(fut_claims/exp_future, bm_pred_rate)
ae_emod  = ae_mae_by_band(fut_claims/exp_future, emod_pred_rate)

print("=" * 68)
print(f"{'Metric':<38} {'Flat Rate':>10} {'NCD/BM':>10} {'E-Mod':>10}")
print("=" * 68)
rows = [
    ("Gini vs true risk (higher=better)",    gini_flat,  gini_ncd,  gini_emod),
    ("RMSE vs true rate (lower=better)",     rmse_flat,  rmse_ncd,  rmse_emod),
    ("A/E MAE by band (lower=better)",       ae_flat,    ae_ncd,    ae_emod),
    ("Scoring time (s)",                      base_time,  ncd_time,  emod_time),
]
for name, b, ncd, em in rows:
    print(f"{name:<38} {b:>10.4f} {ncd:>10.4f} {em:>10.4f}")
print("=" * 68)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Steady-State Distribution and Claiming Threshold

# COMMAND ----------

# Show BM simulator — what does the NCD scale converge to?
sim = BonusMalusSimulator(scale, claim_frequency=BASE_RATE, rng_seed=42)
steady = sim.stationary_distribution(method="analytical")
expected_pf = sim.expected_premium_factor()

print("Steady-state NCD distribution:")
print(steady.to_pandas().to_string(index=False))
print(f"\nExpected premium factor at steady state: {expected_pf:.4f}")
print(f"  (pure premium factor relative to base rate with no NCD)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Plots

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

# Plot 1: Lorenz curves
def lorenz(y, yhat, w=None):
    if w is None: w = np.ones_like(y)
    order = np.argsort(yhat)
    ys, ws = y[order], w[order]
    cw = np.cumsum(ws)/ws.sum(); cy = np.cumsum(ys*ws)/(ys*ws).sum()
    return cw, cy

diag = np.linspace(0,1,100)
cw_f, cy_f = lorenz(true_future_rate, flat_factor*BASE_RATE, exp_future)
cw_n, cy_n = lorenz(true_future_rate, bm_pred_rate,          exp_future)
cw_e, cy_e = lorenz(true_future_rate, emod_pred_rate,        exp_future)

ax1.plot(diag, diag, "k--", alpha=0.5, label="Random (Gini=0)")
ax1.plot(cw_f, cy_f, "grey", linewidth=2, label=f"Flat rate (Gini={gini_flat:.3f})")
ax1.plot(cw_n, cy_n, "b-",  linewidth=2, label=f"NCD/BM (Gini={gini_ncd:.3f})")
ax1.plot(cw_e, cy_e, "r-",  linewidth=2, label=f"E-Mod (Gini={gini_emod:.3f})")
ax1.set_xlabel("Cumul. policies (sorted by predicted rate)"); ax1.set_ylabel("Cumul. true risk")
ax1.set_title("Lorenz Curve: discrimination vs true risk"); ax1.legend(); ax1.grid(True, alpha=0.3)

# Plot 2: Predicted vs true rate scatter (E-Mod)
ax2.scatter(true_future_rate[:500], emod_pred_rate[:500], alpha=0.4, s=10, c="tomato", label=f"E-Mod (RMSE={rmse_emod:.4f})")
ax2.scatter(true_future_rate[:500], bm_pred_rate[:500],   alpha=0.4, s=10, c="steelblue", label=f"NCD (RMSE={rmse_ncd:.4f})")
lim = [0, true_future_rate[:500].max()*1.1]
ax2.plot(lim, lim, "k--", linewidth=1.5)
ax2.set_xlabel("True latent rate"); ax2.set_ylabel("Predicted rate")
ax2.set_title("Predicted vs true rate (500 risks)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Plot 3: A/E by claims history quintile
hist_tot = hist_claims.sum(axis=1)
claim_q = pd.qcut(hist_tot, 5, labels=False, duplicates="drop")
ae_by_hist_flat, ae_by_hist_ncd, ae_by_hist_emod = [], [], []
for q in range(5):
    m = claim_q == q
    if m.sum() < 5:
        ae_by_hist_flat.append(np.nan); ae_by_hist_ncd.append(np.nan); ae_by_hist_emod.append(np.nan)
        continue
    def ae_q(pred):
        return float((fut_claims[m]*exp_future[m]).sum() / max((pred[m]*exp_future[m]).sum(), 1e-10))
    ae_by_hist_flat.append(ae_q(flat_factor*BASE_RATE))
    ae_by_hist_ncd.append(ae_q(bm_pred_rate))
    ae_by_hist_emod.append(ae_q(emod_pred_rate))

xq = np.arange(1, 6)
ax3.bar(xq-0.3, ae_by_hist_flat, 0.3, label="Flat", color="grey", alpha=0.8)
ax3.bar(xq,     ae_by_hist_ncd,  0.3, label="NCD",  color="steelblue", alpha=0.8)
ax3.bar(xq+0.3, ae_by_hist_emod, 0.3, label="E-Mod", color="tomato", alpha=0.8)
ax3.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
ax3.set_xlabel("Claims history quintile (1=fewest claims)"); ax3.set_ylabel("A/E ratio")
ax3.set_title("A/E by claims history band"); ax3.legend(fontsize=8); ax3.set_xticks(xq)
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Steady-state BM distribution
ax4.bar([lvl.ncd_percent for lvl in scale.levels],
        steady["stationary_prob"].to_list(),
        width=3, color="steelblue", alpha=0.8)
ax4.set_xlabel("NCD %"); ax4.set_ylabel("Stationary probability")
ax4.set_title(f"BM steady-state distribution\n(freq={BASE_RATE:.2f}, E[premium factor]={expected_pf:.3f})")
ax4.grid(True, alpha=0.3, axis="y")

# Plot 5: Credibility weight by exposure
exp_range = np.linspace(0.2, 6.0, 100)
omega_curve = exp_range / (exp_range + kappa_hat)
ax5.plot(exp_range, omega_curve, "r-", linewidth=2.5, label=f"omega(e) = e/(e+{kappa_hat:.2f})")
ax5.scatter(exposures.sum(axis=1), A_cred, alpha=0.15, s=8, color="steelblue", label="Individual risks")
ax5.axhline(0.5, color="grey", linestyle="--", alpha=0.6, label="50% credibility")
ax5.set_xlabel("Total exposure (veh-years)"); ax5.set_ylabel("Credibility weight")
ax5.set_title(f"E-Mod credibility weight\nkappa={kappa_hat:.3f}  (3yr avg={credibility_at_3yr:.0%})"); ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3); ax5.set_ylim(0, 1.05)

plt.suptitle("experience-rating: NCD/E-Mod vs Flat Portfolio Rate",
             fontsize=13, fontweight="bold")
plt.savefig("/tmp/benchmark_experience_rating.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_experience_rating.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verdict

# COMMAND ----------

print("=" * 62)
print("VERDICT: Experience Rating vs Flat Portfolio Rate")
print("=" * 62)
print()
print(f"Portfolio: {N_RISKS:,} risks, {N_YEARS_HIST}yr history, alpha={ALPHA_GAMMA} (heterogeneity)")
print(f"Estimated kappa: {kappa_hat:.3f}  =>  3yr credibility: {credibility_at_3yr:.0%}")
print()
print(f"{'Method':<20} {'Gini':>10} {'RMSE':>10} {'A/E MAE':>10}")
print("-" * 54)
for name, g, r, a in [
    ("Flat rate",    gini_flat,  rmse_flat,  ae_flat),
    ("NCD/BM",       gini_ncd,   rmse_ncd,   ae_ncd),
    ("E-Mod",        gini_emod,  rmse_emod,  ae_emod),
]:
    print(f"  {name:<18} {g:>10.4f} {r:>10.4f} {a:>10.4f}")
print()

best_gini = np.argmax([gini_flat, gini_ncd, gini_emod])
best_rmse = np.argmin([rmse_flat, rmse_ncd, rmse_emod])
methods = ["Flat", "NCD", "E-Mod"]
print(f"Best discrimination:   {methods[best_gini]}")
print(f"Best rate accuracy:    {methods[best_rmse]}")
print()
print("Key insight:")
print(f"  NCD improves on flat rate but uses binary claim thresholds")
print(f"  without exposure weighting. A policy with 0.5yr history")
print(f"  gets the same NCD credit as one with 1.5yr history.")
print(f"  E-Mod uses credibility weight omega={credibility_at_3yr:.0%} at 3yr")
print(f"  exposure — more weight on experience for large, stable risks.")

if __name__ == "__main__":
    pass
