"""
Experience modification factors and schedule rating.

Experience modification factors (e-mods) adjust a risk's premium based on its
own loss history relative to what was expected for a risk of its type. The
formula is a credibility-weighted blend of actual and expected losses, with a
ballast term that limits the influence of individual large losses.

This is the NCCI-style approach used in US workers' compensation and adapted
for UK commercial lines. The UK market more often uses bespoke credibility
schedules, but the underlying mathematics is identical.

Schedule rating is a separate (older) technique: underwriters apply
judgemental debits and credits within pre-approved bounds for factors not
captured in the statistical rate. This module provides a validated container
for schedule rating factors.

Design decisions:

- ExperienceModFactor uses the standard formula: Mod = (A * actual + (1-A) *
  expected + B) / (expected + B). This is parameterised by credibility weight A
  and ballast B. We expose these directly rather than hiding them behind a
  calibration API, because practitioners need to understand and defend the
  parameters.

- Batch calculation operates on Polars DataFrames. This is the right choice for
  portfolio-level repricing; you typically have thousands of risks to mod at once.

- ScheduleRating validates debits and credits at entry time. Silent out-of-range
  factors are a compliance risk in regulated markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# ExperienceModFactor
# ---------------------------------------------------------------------------


@dataclass
class CredibilityParams:
    """Parameters for the experience modification formula.

    The formula is:
        Mod = (A * actual + (1 - A) * expected + B) / (expected + B)

    Where:
        A = credibility weight (0 to 1)
        B = ballast value (limits sensitivity to individual large losses)

    Attributes:
        credibility_weight: A in the formula. Typically derived from exposure
            relative to a full-credibility threshold.
        ballast: B in the formula. Controls the sensitivity of the mod to a
            risk's own experience. The sensitivity is A * E / (E + B), where E
            is expected losses. For A=0.65 and B=8,000: a fleet with E=£8k has
            sensitivity 32.5%; at E=£80k sensitivity is ~59%. Larger fleets get
            more weight on their own experience automatically.
    """

    credibility_weight: float
    ballast: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.credibility_weight <= 1.0:
            raise ValueError(
                f"credibility_weight must be in [0, 1], got {self.credibility_weight}."
            )
        if self.ballast < 0:
            raise ValueError(f"ballast must be non-negative, got {self.ballast}.")


class ExperienceModFactor:
    """Experience modification factor calculator.

    Computes experience modification factors using the credibility-weighted
    formula standard in commercial lines experience rating.

    Formula:
        Mod = (A * actual_losses + (1 - A) * expected_losses + B) / (expected_losses + B)

    A Mod > 1.0 means the risk has performed worse than expected; < 1.0 means
    better than expected.

    Example::

        params = CredibilityParams(credibility_weight=0.70, ballast=5000.0)
        emod = ExperienceModFactor(params)

        # Single risk
        mod = emod.predict(expected_losses=20_000, actual_losses=25_000)

        # Batch (Polars DataFrame)
        df = pl.DataFrame({
            "risk_id": ["A", "B", "C"],
            "expected_losses": [20_000.0, 50_000.0, 10_000.0],
            "actual_losses": [25_000.0, 40_000.0, 8_000.0],
        })
        result = emod.predict_batch(df)
    """

    def __init__(self, params: CredibilityParams) -> None:
        """Initialise with credibility parameters.

        Args:
            params: CredibilityParams defining A and B in the mod formula.
        """
        self.params = params

    @classmethod
    def from_exposure(
        cls,
        actual_exposure: float,
        full_credibility_exposure: float,
        ballast: float,
        credibility_formula: str = "square_root",
    ) -> "ExperienceModFactor":
        """Construct from exposure-based credibility.

        Common in practice: credibility A is derived from the ratio of actual
        exposure to the exposure needed for full credibility.

        Args:
            actual_exposure: Policyholder's actual exposure (e.g., payroll,
                vehicle-years, floor area).
            full_credibility_exposure: Exposure at which A = 1.0.
            ballast: Ballast B in the mod formula.
            credibility_formula: "square_root" (classic Mowbray formula,
                A = sqrt(min(e / e_full, 1.0)), capped at 1.0) or "linear"
                (A = min(e / e_full, 1.0), capped at 1.0).

        Returns:
            ExperienceModFactor: Configured with derived credibility weight.
        """
        ratio = actual_exposure / full_credibility_exposure
        if credibility_formula == "square_root":
            A = float(np.sqrt(min(ratio, 1.0)))
        elif credibility_formula == "linear":
            A = float(min(ratio, 1.0))
        else:
            raise ValueError(
                f"Unknown credibility_formula '{credibility_formula}'. "
                "Use 'square_root' or 'linear'."
            )
        return cls(CredibilityParams(credibility_weight=A, ballast=ballast))

    def predict(
        self,
        expected_losses: float,
        actual_losses: float,
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> float:
        """Compute experience modification factor for a single risk.

        Args:
            expected_losses: Expected losses for the risk (£).
            actual_losses: Actual incurred losses for the risk (£).
            cap: Maximum allowed mod factor (e.g., 2.0). None = no cap.
            floor: Minimum allowed mod factor (e.g., 0.5). None = no floor.

        Returns:
            float: Modification factor. 1.0 = no adjustment.

        Raises:
            ValueError: If expected_losses is zero or negative.
        """
        if expected_losses <= 0:
            raise ValueError(
                f"expected_losses must be positive, got {expected_losses}."
            )
        if actual_losses < 0:
            raise ValueError(
                f"actual_losses must be non-negative, got {actual_losses}."
            )

        A = self.params.credibility_weight
        B = self.params.ballast

        mod = (A * actual_losses + (1 - A) * expected_losses + B) / (expected_losses + B)

        if cap is not None:
            mod = min(mod, cap)
        if floor is not None:
            mod = max(mod, floor)

        return float(mod)

    def predict_batch(
        self,
        df: pl.DataFrame,
        expected_col: str = "expected_losses",
        actual_col: str = "actual_losses",
        cap: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> pl.DataFrame:
        """Compute experience modification factors for a portfolio of risks.

        Args:
            df: Polars DataFrame containing expected and actual loss columns.
            expected_col: Name of the expected losses column.
            actual_col: Name of the actual losses column.
            cap: Maximum mod factor. None = no cap.
            floor: Minimum mod factor. None = no floor.

        Returns:
            pl.DataFrame: Input DataFrame with an additional "mod_factor" column.

        Raises:
            ValueError: If required columns are missing or if expected_losses
                contains non-positive values.
        """
        missing = {expected_col, actual_col} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if (df[expected_col] <= 0).any():
            raise ValueError("All expected_losses values must be positive.")

        if (df[actual_col] < 0).any():
            raise ValueError("All actual_losses values must be non-negative.")

        A = self.params.credibility_weight
        B = self.params.ballast

        mod = (
            (A * pl.col(actual_col) + (1 - A) * pl.col(expected_col) + B)
            / (pl.col(expected_col) + B)
        )

        result = df.with_columns(mod.alias("mod_factor"))

        if cap is not None:
            result = result.with_columns(
                pl.col("mod_factor").clip(upper_bound=cap)
            )
        if floor is not None:
            result = result.with_columns(
                pl.col("mod_factor").clip(lower_bound=floor)
            )

        return result

    def sensitivity(
        self,
        expected_losses: float,
        actual_range: Optional[tuple[float, float]] = None,
        n_points: int = 50,
    ) -> pl.DataFrame:
        """Return mod factors across a range of actual losses.

        Useful for showing underwriters how the mod factor responds to
        different loss outcomes.

        Args:
            expected_losses: Expected losses (fixed).
            actual_range: (min_actual, max_actual) range. Defaults to
                (0, 3 * expected_losses).
            n_points: Number of points in the range.

        Returns:
            pl.DataFrame: Columns: actual_losses, mod_factor, loss_ratio
                (actual / expected).
        """
        if actual_range is None:
            actual_range = (0.0, 3.0 * expected_losses)

        actuals = np.linspace(actual_range[0], actual_range[1], n_points)
        mods = [self.predict(expected_losses, a) for a in actuals]

        return pl.DataFrame({
            "actual_losses": actuals.tolist(),
            "mod_factor": mods,
            "loss_ratio": (actuals / expected_losses).tolist(),
        })


# ---------------------------------------------------------------------------
# ScheduleRating
# ---------------------------------------------------------------------------


@dataclass
class ScheduleFactor:
    """A single factor in a schedule rating system.

    Attributes:
        name: Factor name (e.g., "Premises condition").
        description: Human-readable description of what this factor measures.
        min_credit: Maximum credit (negative debit). Must be <= 0.
        max_debit: Maximum debit. Must be >= 0.
        applied_value: The debit/credit actually applied for this risk.
            Positive = debit (premium increase), negative = credit (decrease).
    """

    name: str
    description: str
    min_credit: float  # <= 0
    max_debit: float   # >= 0
    applied_value: float = 0.0

    def __post_init__(self) -> None:
        if self.min_credit > 0:
            raise ValueError(
                f"Factor '{self.name}': min_credit must be <= 0, "
                f"got {self.min_credit}."
            )
        if self.max_debit < 0:
            raise ValueError(
                f"Factor '{self.name}': max_debit must be >= 0, "
                f"got {self.max_debit}."
            )


class ScheduleRating:
    """Schedule rating system with validated debit/credit factors.

    Schedule rating allows underwriters to apply judgemental adjustments within
    pre-approved bounds. The total adjustment is the sum of all debit/credit
    factors (additive convention), capped at the schedule maximum. The final
    factor returned is 1.0 + total_adjustment.

    In the UK commercial market, schedule rating is more common in liability,
    property, and professional indemnity than in personal lines. The bounds are
    typically set by underwriting authority and validated against market guidance.

    Example::

        sr = ScheduleRating(max_total_debit=0.25, max_total_credit=0.25)
        sr.add_factor("Premises", "Quality and maintenance of premises", -0.10, 0.10)
        sr.add_factor("Management", "Management quality and experience", -0.05, 0.05)

        factor = sr.rate({"Premises": 0.05, "Management": -0.03})
        print(f"Schedule rating factor: {factor:.4f}")
    """

    def __init__(
        self,
        max_total_debit: float = 0.25,
        max_total_credit: float = 0.25,
    ) -> None:
        """Initialise the schedule rating system.

        Args:
            max_total_debit: Maximum aggregate debit as a proportion (e.g.,
                0.25 = 25% maximum debit). Applied as a cap on the final factor.
            max_total_credit: Maximum aggregate credit as a proportion.
        """
        if max_total_debit < 0:
            raise ValueError("max_total_debit must be non-negative.")
        if max_total_credit < 0:
            raise ValueError("max_total_credit must be non-negative.")

        self.max_total_debit = max_total_debit
        self.max_total_credit = max_total_credit
        self._factors: dict[str, ScheduleFactor] = {}

    def add_factor(
        self,
        name: str,
        min_credit: float,
        max_debit: float,
        description: str = "",
    ) -> "ScheduleRating":
        """Add a rating factor to the schedule.

        Args:
            name: Unique factor name.
            min_credit: Maximum credit (e.g., -0.10 = up to 10% credit).
            max_debit: Maximum debit (e.g., 0.10 = up to 10% debit).
            description: Human-readable description.

        Returns:
            ScheduleRating: Self, for chaining.

        Raises:
            ValueError: If bounds are invalid or name already exists.
        """
        if name in self._factors:
            raise ValueError(f"Factor '{name}' already exists.")

        self._factors[name] = ScheduleFactor(
            name=name,
            description=description,
            min_credit=min_credit,
            max_debit=max_debit,
        )
        return self

    def rate(self, features: dict[str, float]) -> float:
        """Apply schedule rating factors for a specific risk.

        Args:
            features: Mapping of factor name to applied debit/credit value.
                Positive = debit. Negative = credit. Must be within the
                factor's [min_credit, max_debit] bounds.

        Returns:
            float: Multiplicative schedule rating factor. Values > 1.0 increase
                premium; < 1.0 decrease it.

        Raises:
            ValueError: If a factor name is unrecognised or value is out of bounds.
        """
        for name, value in features.items():
            if name not in self._factors:
                raise ValueError(
                    f"Unknown factor '{name}'. Add it first with add_factor()."
                )
            factor = self._factors[name]
            if value < factor.min_credit - 1e-10:
                raise ValueError(
                    f"Factor '{name}': value {value} is below min_credit "
                    f"{factor.min_credit}."
                )
            if value > factor.max_debit + 1e-10:
                raise ValueError(
                    f"Factor '{name}': value {value} exceeds max_debit "
                    f"{factor.max_debit}."
                )

        # Additive schedule: sum all debits/credits, then apply
        total_adjustment = sum(features.values())

        # Apply aggregate bounds
        total_adjustment = max(total_adjustment, -self.max_total_credit)
        total_adjustment = min(total_adjustment, self.max_total_debit)

        return 1.0 + total_adjustment

    def rate_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply schedule rating to a portfolio.

        Args:
            df: Polars DataFrame where each column corresponds to a factor
                name and contains the applied debit/credit values for each risk.
                Only columns that match registered factor names are used.

        Returns:
            pl.DataFrame: Input DataFrame with an additional "schedule_factor" column.
        """
        factor_cols = [c for c in df.columns if c in self._factors]
        if not factor_cols:
            raise ValueError(
                "No recognised factor columns found in DataFrame. "
                f"Registered factors: {list(self._factors.keys())}"
            )

        # Validate bounds per column
        for col in factor_cols:
            factor = self._factors[col]
            out_of_range = df.filter(
                (pl.col(col) < factor.min_credit - 1e-10)
                | (pl.col(col) > factor.max_debit + 1e-10)
            )
            if len(out_of_range) > 0:
                raise ValueError(
                    f"Factor '{col}' has {len(out_of_range)} out-of-range values."
                )

        # Sum all factor columns
        total = pl.sum_horizontal([pl.col(c) for c in factor_cols])
        # Clip to aggregate bounds
        total_clipped = total.clip(
            lower_bound=-self.max_total_credit,
            upper_bound=self.max_total_debit,
        )

        return df.with_columns((1.0 + total_clipped).alias("schedule_factor"))

    def summary(self) -> pl.DataFrame:
        """Return a summary of registered factors.

        Returns:
            pl.DataFrame: One row per factor with name, description,
                min_credit, max_debit.
        """
        if not self._factors:
            return pl.DataFrame(
                schema={
                    "name": pl.Utf8,
                    "description": pl.Utf8,
                    "min_credit": pl.Float64,
                    "max_debit": pl.Float64,
                }
            )
        rows = [
            {
                "name": f.name,
                "description": f.description,
                "min_credit": f.min_credit,
                "max_debit": f.max_debit,
            }
            for f in self._factors.values()
        ]
        return pl.DataFrame(rows)

    @property
    def factor_names(self) -> list[str]:
        """Names of all registered factors."""
        return list(self._factors.keys())

    def __repr__(self) -> str:
        return (
            f"ScheduleRating("
            f"n_factors={len(self._factors)}, "
            f"max_debit={self.max_total_debit}, "
            f"max_credit={self.max_total_credit})"
        )
