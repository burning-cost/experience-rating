"""
Bonus-malus / NCD (no-claims discount) systems for motor insurance.

The standard UK motor NCD system works as a Markov chain: policyholders move
between levels based on whether they make claims in each policy year. This
module provides tools to define arbitrary BM scales, simulate policyholder
flows, compute steady-state distributions, and determine optimal claiming
thresholds.

Design decisions:

- We represent transition rules as a dict mapping claim count -> next level.
  This handles both simple "one step up / two steps down" rules and more
  exotic commercial structures.

- Monte Carlo simulation is done in NumPy (vectorised) rather than row-by-row
  Python loops. On 100k policyholders x 20 years this is fast enough to run
  interactively.

- Stationary distribution is computed both analytically (left eigenvector of
  the transition matrix for eigenvalue 1) and via simulation. The two methods
  should agree to within simulation noise; if they don't, that's a bug.

- ClaimThreshold uses a finite-horizon NPV comparison: "cost of claiming"
  (NCD lost x premium over horizon) vs claim amount. No utility theory, just
  straightforward financial arithmetic. Practitioners want a number they can
  explain to a customer, not a utility maximisation argument.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl
from scipy.linalg import eig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TransitionRules:
    """Transition rules from a given BM level.

    Attributes:
        claim_free_level: Level to move to if no claims in period.
        claim_levels: Mapping from number-of-claims (1, 2, 3, ...) to
            resulting level. If a claim count exceeds the highest key, the
            lowest level (worst) in the mapping is used.
    """

    claim_free_level: int
    claim_levels: dict[int, int]

    def next_level(self, n_claims: int) -> int:
        """Return the next BM level given a number of claims.

        Args:
            n_claims: Number of claims in the period (0 = claim-free).

        Returns:
            Next BM level index.
        """
        if n_claims == 0:
            return self.claim_free_level
        if n_claims in self.claim_levels:
            return self.claim_levels[n_claims]
        # More claims than we have explicit rules for: use worst outcome
        return min(self.claim_levels.values())


@dataclass
class BonusMalusLevel:
    """A single level in a bonus-malus scale.

    Attributes:
        index: Zero-based index within the scale (0 = worst/highest premium).
        name: Human-readable name, e.g. "0% NCD" or "Step 5".
        premium_factor: Multiplicative premium relativity at this level.
            1.0 = no discount/loading. 0.65 = 35% discount.
        transitions: Rules for moving from this level.
        ncd_percent: NCD percentage equivalent (optional, for display).
    """

    index: int
    name: str
    premium_factor: float
    transitions: TransitionRules
    ncd_percent: Optional[int] = None


# ---------------------------------------------------------------------------
# BonusMalusScale
# ---------------------------------------------------------------------------


class BonusMalusScale:
    """A complete bonus-malus / NCD scale.

    A BM scale is a finite set of premium levels and the rules that govern
    transitions between them. This class validates the scale and provides
    utilities for computing the transition matrix.

    Example::

        scale = BonusMalusScale.from_uk_standard()
        print(scale.levels)
        T = scale.transition_matrix(claim_frequency=0.10)
    """

    def __init__(self, levels: list[BonusMalusLevel]) -> None:
        """Initialise with a list of BM levels.

        Args:
            levels: List of BonusMalusLevel objects. Must be ordered by index
                (0, 1, 2, ...) with no gaps.

        Raises:
            ValueError: If the scale is malformed.
        """
        self._validate(levels)
        self.levels = levels
        self._index_map = {lvl.index: lvl for lvl in levels}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, levels: list[BonusMalusLevel]) -> None:
        """Check internal consistency of the scale.

        Args:
            levels: Levels to validate.

        Raises:
            ValueError: If any check fails.
        """
        if not levels:
            raise ValueError("BM scale must have at least one level.")

        indices = [lvl.index for lvl in levels]
        expected = list(range(len(levels)))
        if sorted(indices) != expected:
            raise ValueError(
                f"Level indices must be 0..{len(levels)-1}, got {sorted(indices)}."
            )

        valid_indices = set(expected)
        for lvl in levels:
            tr = lvl.transitions
            if tr.claim_free_level not in valid_indices:
                raise ValueError(
                    f"Level {lvl.index}: claim_free_level "
                    f"{tr.claim_free_level} is not a valid index."
                )
            for n_claims, dest in tr.claim_levels.items():
                if dest not in valid_indices:
                    raise ValueError(
                        f"Level {lvl.index}: transition for {n_claims} claim(s) "
                        f"leads to invalid level {dest}."
                    )
            if lvl.premium_factor <= 0:
                raise ValueError(
                    f"Level {lvl.index}: premium_factor must be positive, "
                    f"got {lvl.premium_factor}."
                )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_uk_standard(cls) -> "BonusMalusScale":
        """ABI-style standard UK motor NCD scale.

        10 levels (0%-65% NCD). One claim-free year moves up one level.
        One claim moves back two levels (minimum level 0). Two or more claims
        in a year move back to level 0.

        The premium factors represent typical market relativities for
        illustration purposes. Real insurers set their own relativities.

        Returns:
            BonusMalusScale: Standard UK 10-level NCD scale.
        """
        # NCD percentages at each level (0 = no NCD, 9 = 65% NCD)
        ncd_pcts = [0, 10, 20, 30, 40, 45, 50, 55, 60, 65]
        premium_factors = [1 - p / 100 for p in ncd_pcts]

        n = len(ncd_pcts)
        levels = []
        for i in range(n):
            # Claim-free: move up one level (cap at max)
            claim_free = min(i + 1, n - 1)
            # One claim: move back two levels (floor at 0)
            one_claim = max(i - 2, 0)
            # Two or more claims: drop to level 0
            two_plus = 0

            transitions = TransitionRules(
                claim_free_level=claim_free,
                claim_levels={1: one_claim, 2: two_plus, 3: two_plus},
            )
            levels.append(
                BonusMalusLevel(
                    index=i,
                    name=f"{ncd_pcts[i]}% NCD",
                    premium_factor=premium_factors[i],
                    transitions=transitions,
                    ncd_percent=ncd_pcts[i],
                )
            )
        return cls(levels)

    @classmethod
    def from_dict(cls, spec: dict) -> "BonusMalusScale":
        """Build a BM scale from a dictionary specification.

        The dict format::

            {
                "levels": [
                    {
                        "index": 0,
                        "name": "Step 0",
                        "premium_factor": 1.20,
                        "ncd_percent": null,
                        "transitions": {
                            "claim_free_level": 1,
                            "claim_levels": {"1": 0, "2": 0}
                        }
                    },
                    ...
                ]
            }

        Args:
            spec: Dictionary matching the format above.

        Returns:
            BonusMalusScale: Constructed scale.
        """
        levels = []
        for lvl_spec in spec["levels"]:
            tr_spec = lvl_spec["transitions"]
            transitions = TransitionRules(
                claim_free_level=tr_spec["claim_free_level"],
                claim_levels={int(k): v for k, v in tr_spec["claim_levels"].items()},
            )
            levels.append(
                BonusMalusLevel(
                    index=lvl_spec["index"],
                    name=lvl_spec["name"],
                    premium_factor=lvl_spec["premium_factor"],
                    ncd_percent=lvl_spec.get("ncd_percent"),
                    transitions=transitions,
                )
            )
        return cls(levels)

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def transition_matrix(
        self,
        claim_frequency: float,
        max_claims_per_period: int = 3,
    ) -> np.ndarray:
        """Compute the Markov transition matrix.

        Assumes claims follow a Poisson process with the given frequency.
        Entry T[i, j] is the probability of moving from level i to level j.

        Args:
            claim_frequency: Expected claims per policyholder per year (lambda
                of the Poisson distribution). Typical UK motor: 0.05 - 0.15.
            max_claims_per_period: Maximum number of claims to model explicitly.
                Higher values are collapsed into this bin.

        Returns:
            np.ndarray: n x n row-stochastic transition matrix.
        """
        n = len(self.levels)
        T = np.zeros((n, n))

        from scipy.stats import poisson

        # Precompute Poisson probabilities up to max_claims_per_period
        probs = np.array([
            poisson.pmf(k, claim_frequency) for k in range(max_claims_per_period)
        ])
        prob_max_plus = 1.0 - probs.sum()  # P(claims >= max_claims_per_period)

        for lvl in self.levels:
            i = lvl.index
            # k = 0, 1, ..., max_claims_per_period-1
            for k in range(max_claims_per_period):
                dest = lvl.transitions.next_level(k)
                T[i, dest] += probs[k]
            # k >= max_claims_per_period
            dest = lvl.transitions.next_level(max_claims_per_period)
            T[i, dest] += prob_max_plus

        return T

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> pl.DataFrame:
        """Return a Polars DataFrame summarising the scale.

        Returns:
            pl.DataFrame: One row per level with index, name, premium_factor,
                ncd_percent, claim_free_destination, one_claim_destination.
        """
        rows = []
        for lvl in self.levels:
            rows.append({
                "index": lvl.index,
                "name": lvl.name,
                "premium_factor": lvl.premium_factor,
                "ncd_percent": lvl.ncd_percent,
                "claim_free_dest": lvl.transitions.claim_free_level,
                "one_claim_dest": lvl.transitions.claim_levels.get(1, 0),
            })
        return pl.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.levels)

    def __repr__(self) -> str:
        return f"BonusMalusScale(n_levels={len(self.levels)})"


# ---------------------------------------------------------------------------
# BonusMalusSimulator
# ---------------------------------------------------------------------------


class BonusMalusSimulator:
    """Monte Carlo simulation of policyholder flows through a BM scale.

    Given a BM scale and a claim frequency assumption, this class simulates
    individual policyholders moving between levels over time and computes
    steady-state (long-run) distributions.

    The analytical stationary distribution is computed from the eigenvector
    of the transition matrix corresponding to eigenvalue 1. The simulation
    should converge to this as n_years -> infinity.

    Example::

        scale = BonusMalusScale.from_uk_standard()
        sim = BonusMalusSimulator(scale, claim_frequency=0.10)
        result = sim.simulate(n_policyholders=10_000, n_years=20)
        stat = sim.stationary_distribution()
    """

    def __init__(
        self,
        scale: BonusMalusScale,
        claim_frequency: float,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Initialise the simulator.

        Args:
            scale: The BM scale to simulate.
            claim_frequency: Expected claims per policyholder per year.
            rng_seed: Random seed for reproducibility.
        """
        if claim_frequency < 0:
            raise ValueError(
                f"claim_frequency must be non-negative, got {claim_frequency}."
            )
        self.scale = scale
        self.claim_frequency = claim_frequency
        self._rng = np.random.default_rng(rng_seed)
        self._T: Optional[np.ndarray] = None

    @property
    def transition_matrix(self) -> np.ndarray:
        """Cached transition matrix (computed on first access)."""
        if self._T is None:
            self._T = self.scale.transition_matrix(self.claim_frequency)
        return self._T

    def simulate(
        self,
        n_policyholders: int = 10_000,
        n_years: int = 20,
        starting_level: Optional[int] = None,
    ) -> pl.DataFrame:
        """Simulate policyholder flows through the BM scale.

        All policyholders start at the same level (default: level 0, i.e., no
        existing NCD) and progress through the scale over n_years.

        Args:
            n_policyholders: Number of policyholders to simulate.
            n_years: Number of policy years to simulate.
            starting_level: Starting BM level index. Defaults to 0 (new
                business, no NCD).

        Returns:
            pl.DataFrame: Columns: year (0..n_years), level (0..n-1),
                count (number of policyholders at that level in that year),
                proportion (count / n_policyholders),
                premium_factor (scale premium relativity at that level).
        """
        n_levels = len(self.scale)
        if starting_level is None:
            starting_level = 0
        if starting_level < 0 or starting_level >= n_levels:
            raise ValueError(
                f"starting_level {starting_level} out of range [0, {n_levels-1}]."
            )

        # Current level for each policyholder
        current = np.full(n_policyholders, starting_level, dtype=np.int32)

        # Pre-build lookup: for each level, given claim count -> next level
        # We'll generate Poisson claims and vectorise the transition
        transition_lookup = self._build_lookup(n_levels)

        rows = []
        # Record year 0 (starting state)
        counts = np.bincount(current, minlength=n_levels)
        for lvl in range(n_levels):
            rows.append({
                "year": 0,
                "level": lvl,
                "count": int(counts[lvl]),
                "proportion": float(counts[lvl]) / n_policyholders,
                "premium_factor": self.scale.levels[lvl].premium_factor,
            })

        for year in range(1, n_years + 1):
            # Draw claims from Poisson
            claims = self._rng.poisson(self.claim_frequency, size=n_policyholders)
            claims_capped = np.minimum(claims, transition_lookup.shape[1] - 1)
            # Vectorised lookup
            current = transition_lookup[current, claims_capped]

            counts = np.bincount(current, minlength=n_levels)
            for lvl in range(n_levels):
                rows.append({
                    "year": year,
                    "level": lvl,
                    "count": int(counts[lvl]),
                    "proportion": float(counts[lvl]) / n_policyholders,
                    "premium_factor": self.scale.levels[lvl].premium_factor,
                })

        return pl.DataFrame(rows)

    def _build_lookup(self, n_levels: int, max_claims: int = 5) -> np.ndarray:
        """Build a [n_levels x (max_claims+1)] integer lookup table.

        Entry [i, k] = next level from level i given k claims.

        Args:
            n_levels: Number of levels in the scale.
            max_claims: Maximum claim count to represent explicitly.

        Returns:
            np.ndarray: Integer array of shape (n_levels, max_claims+1).
        """
        lookup = np.zeros((n_levels, max_claims + 1), dtype=np.int32)
        for lvl in self.scale.levels:
            i = lvl.index
            for k in range(max_claims + 1):
                lookup[i, k] = lvl.transitions.next_level(k)
        return lookup

    def stationary_distribution(
        self, method: str = "analytical"
    ) -> pl.DataFrame:
        """Compute the steady-state distribution over BM levels.

        Args:
            method: "analytical" (eigenvector of transition matrix) or
                "simulation" (long-run average of simulate()).

        Returns:
            pl.DataFrame: Columns: level, name, stationary_prob,
                premium_factor.

        Raises:
            ValueError: If method is not recognised.
        """
        if method == "analytical":
            return self._stationary_analytical()
        elif method == "simulation":
            return self._stationary_simulation()
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'analytical' or 'simulation'.")

    def _stationary_analytical(self) -> pl.DataFrame:
        """Compute stationary distribution via left eigenvector.

        The stationary distribution pi satisfies pi @ T = pi (row-stochastic T).
        Equivalently, pi is the left eigenvector for eigenvalue 1, which is
        the right eigenvector of T.T for eigenvalue 1.

        Returns:
            pl.DataFrame: Stationary distribution.
        """
        T = self.transition_matrix
        # Eigenvalues/vectors of T^T; we want left eigenvectors of T
        eigenvalues, eigenvectors = eig(T.T)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = eigenvectors[:, idx].real
        pi = pi / pi.sum()  # normalise

        if np.any(pi < -1e-8):
            warnings.warn(
                "Stationary distribution has negative entries; transition matrix "
                "may not be ergodic."
            )
        pi = np.maximum(pi, 0.0)
        pi = pi / pi.sum()

        return pl.DataFrame({
            "level": [lvl.index for lvl in self.scale.levels],
            "name": [lvl.name for lvl in self.scale.levels],
            "stationary_prob": pi.tolist(),
            "premium_factor": [lvl.premium_factor for lvl in self.scale.levels],
        })

    def _stationary_simulation(
        self, n_policyholders: int = 50_000, n_years: int = 100
    ) -> pl.DataFrame:
        """Estimate stationary distribution by simulation.

        Runs a long simulation and averages the distribution across the final
        20 years (to avoid start-up transient effects).

        Args:
            n_policyholders: Simulation size.
            n_years: Total years to simulate.

        Returns:
            pl.DataFrame: Estimated stationary distribution.
        """
        df = self.simulate(n_policyholders=n_policyholders, n_years=n_years)
        # Average over final 20 years
        cutoff = n_years - 20
        steady = (
            df.filter(pl.col("year") > cutoff)
            .group_by("level")
            .agg(pl.mean("proportion").alias("stationary_prob"))
            .sort("level")
        )
        # Join names and premium factors
        meta = pl.DataFrame({
            "level": [lvl.index for lvl in self.scale.levels],
            "name": [lvl.name for lvl in self.scale.levels],
            "premium_factor": [lvl.premium_factor for lvl in self.scale.levels],
        })
        return steady.join(meta, on="level").select(
            ["level", "name", "stationary_prob", "premium_factor"]
        )

    def expected_premium_factor(self, method: str = "analytical") -> float:
        """Expected premium factor at steady state.

        This is the probability-weighted average of premium factors across all
        BM levels in the stationary distribution. It answers: "for a book of
        mature policyholders at this claim frequency, what is the average
        premium factor relative to a no-discount baseline?"

        Args:
            method: Passed to stationary_distribution().

        Returns:
            float: Expected premium factor.
        """
        dist = self.stationary_distribution(method=method)
        return float(
            (dist["stationary_prob"] * dist["premium_factor"]).sum()
        )


# ---------------------------------------------------------------------------
# ClaimThreshold
# ---------------------------------------------------------------------------


class ClaimThreshold:
    """Optimal claiming threshold analysis for NCD-bearing policyholders.

    Should a policyholder claim a small loss, or absorb it out of pocket to
    protect their NCD?

    This class computes the financial break-even: the minimum claim amount at
    which claiming is rational, given the future NCD cost of claiming over a
    chosen time horizon. No utility theory — just NPV arithmetic.

    The analysis depends on:
    - Current NCD level (determines which level they'd fall to after a claim)
    - Annual premium (used to monetise NCD loss)
    - Time horizon (number of years to consider NCD cost over)
    - Discount rate (time value of money)

    Example::

        scale = BonusMalusScale.from_uk_standard()
        ct = ClaimThreshold(scale, discount_rate=0.05)
        threshold = ct.threshold(
            current_level=5,
            annual_premium=800.0,
            years_horizon=3,
        )
        print(f"Claim only if loss exceeds £{threshold:.2f}")
    """

    def __init__(
        self,
        scale: BonusMalusScale,
        discount_rate: float = 0.05,
    ) -> None:
        """Initialise.

        Args:
            scale: The BM scale to analyse.
            discount_rate: Annual discount rate for NPV calculation.
        """
        self.scale = scale
        self.discount_rate = discount_rate

    def _ncd_cost(
        self,
        current_level: int,
        claim_count: int,
        annual_premium: float,
        years_horizon: int,
    ) -> float:
        """Cost (in £) of losing NCD following a claim.

        Computes the NPV of additional premium paid over the recovery period
        compared to the claim-free trajectory.

        Args:
            current_level: Current BM level index.
            claim_count: Number of claims (usually 1).
            annual_premium: Base annual premium (at full rate, i.e., at level 0
                premium_factor of the scale, adjusted to current level).
            years_horizon: Number of years to project forward.

        Returns:
            float: NPV of NCD cost in £.
        """
        # Simulate two trajectories: claim-free and with claim_count claims this year
        scale = self.scale
        n_levels = len(scale)

        def project(starting_level: int, first_year_claims: int) -> list[int]:
            """Project claim-free trajectory after first year."""
            levels = [starting_level]
            # First year: apply first_year_claims
            lvl = scale.levels[starting_level].transitions.next_level(first_year_claims)
            levels.append(lvl)
            # Subsequent years: claim-free (no more claims in this simplified model)
            for _ in range(years_horizon - 1):
                lvl = scale.levels[lvl].transitions.next_level(0)
                levels.append(lvl)
            return levels

        # Trajectory if claim-free this year
        traj_no_claim = project(current_level, 0)
        # Trajectory if claiming this year
        traj_claim = project(current_level, claim_count)

        # Base premium (premium at the policyholder's current level)
        # We interpret annual_premium as the premium the policyholder actually pays,
        # which already incorporates their current NCD.
        # To get the undiscounted base premium: annual_premium / current_factor
        current_factor = scale.levels[current_level].premium_factor
        base_premium = annual_premium / current_factor

        # NPV of premiums under each trajectory (years 1..years_horizon)
        npv_no_claim = sum(
            base_premium * scale.levels[traj_no_claim[t]].premium_factor
            / (1 + self.discount_rate) ** t
            for t in range(1, years_horizon + 1)
        )
        npv_claim = sum(
            base_premium * scale.levels[traj_claim[t]].premium_factor
            / (1 + self.discount_rate) ** t
            for t in range(1, years_horizon + 1)
        )

        return float(npv_claim - npv_no_claim)

    def threshold(
        self,
        current_level: int,
        annual_premium: float,
        years_horizon: int = 3,
        claim_count: int = 1,
    ) -> float:
        """Minimum claim amount that makes claiming financially rational.

        If a loss is smaller than this threshold, the policyholder is better
        off absorbing it out of pocket and protecting their NCD. If the loss
        exceeds the threshold, claiming is rational.

        Args:
            current_level: Current BM level index (0 = no NCD, n-1 = max NCD).
            annual_premium: Premium the policyholder currently pays (£).
            years_horizon: Number of future years to consider NCD cost over.
            claim_count: Number of claims to model (default 1).

        Returns:
            float: Threshold claim amount in £. Claim only if loss > this value.
        """
        if current_level < 0 or current_level >= len(self.scale):
            raise ValueError(
                f"current_level {current_level} out of range "
                f"[0, {len(self.scale)-1}]."
            )
        if years_horizon <= 0:
            raise ValueError(
                f"years_horizon must be a positive integer, got {years_horizon}."
            )
        return self._ncd_cost(current_level, claim_count, annual_premium, years_horizon)

    def threshold_curve(
        self,
        current_level: int,
        annual_premium: float,
        max_horizon: int = 10,
    ) -> pl.DataFrame:
        """Threshold claim amounts for a range of time horizons.

        Returns a DataFrame showing how the claiming threshold changes as the
        time horizon grows. Useful for communicating to customers how their
        NCD horizon assumption affects the decision.

        Args:
            current_level: Current BM level index.
            annual_premium: Premium the policyholder currently pays (£).
            max_horizon: Maximum time horizon in years.

        Returns:
            pl.DataFrame: Columns: years_horizon, threshold_amount,
                current_level, current_ncd_percent.
        """
        rows = []
        for h in range(1, max_horizon + 1):
            th = self.threshold(current_level, annual_premium, years_horizon=h)
            rows.append({
                "years_horizon": h,
                "threshold_amount": th,
                "current_level": current_level,
                "current_ncd_percent": self.scale.levels[current_level].ncd_percent,
                "annual_premium": annual_premium,
            })
        return pl.DataFrame(rows)

    def should_claim(
        self,
        current_level: int,
        claim_amount: float,
        annual_premium: float,
        years_horizon: int = 3,
    ) -> bool:
        """Return True if claiming is financially rational.

        Args:
            current_level: Current BM level index.
            claim_amount: Actual loss amount (£).
            annual_premium: Premium the policyholder currently pays (£).
            years_horizon: Number of future years to consider.

        Returns:
            bool: True if the claim amount exceeds the NCD cost threshold.
        """
        th = self.threshold(current_level, annual_premium, years_horizon)
        return claim_amount > th

    def full_analysis(
        self,
        annual_premium: float,
        years_horizon: int = 3,
    ) -> pl.DataFrame:
        """Compute claiming thresholds for every level in the scale.

        Args:
            annual_premium: Base annual premium before any NCD/BM discount (£).
                Internally multiplied by each level's premium_factor to derive
                the level-specific premium used in the threshold calculation.
            years_horizon: Number of future years to consider.

        Returns:
            pl.DataFrame: One row per level with threshold amounts.
        """
        rows = []
        for lvl in self.scale.levels:
            # Premium at this level
            prem = annual_premium * lvl.premium_factor
            th = self.threshold(lvl.index, prem, years_horizon)
            rows.append({
                "level": lvl.index,
                "name": lvl.name,
                "ncd_percent": lvl.ncd_percent,
                "premium_factor": lvl.premium_factor,
                "annual_premium": prem,
                "claiming_threshold": th,
            })
        return pl.DataFrame(rows)
