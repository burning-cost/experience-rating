"""
Tests for ExperienceModFactor and ScheduleRating.
"""

import numpy as np
import polars as pl
import pytest

from experience_rating.experience_mod import (
    CredibilityParams,
    ExperienceModFactor,
    ScheduleFactor,
    ScheduleRating,
)


# ---------------------------------------------------------------------------
# CredibilityParams
# ---------------------------------------------------------------------------


class TestCredibilityParams:
    def test_valid_params(self):
        p = CredibilityParams(credibility_weight=0.70, ballast=5000.0)
        assert p.credibility_weight == 0.70
        assert p.ballast == 5000.0

    def test_credibility_above_one_raises(self):
        with pytest.raises(ValueError, match="credibility_weight"):
            CredibilityParams(credibility_weight=1.5, ballast=1000.0)

    def test_credibility_below_zero_raises(self):
        with pytest.raises(ValueError, match="credibility_weight"):
            CredibilityParams(credibility_weight=-0.1, ballast=1000.0)

    def test_negative_ballast_raises(self):
        with pytest.raises(ValueError, match="ballast"):
            CredibilityParams(credibility_weight=0.5, ballast=-1.0)

    def test_zero_credibility_allowed(self):
        p = CredibilityParams(credibility_weight=0.0, ballast=0.0)
        assert p.credibility_weight == 0.0

    def test_full_credibility_allowed(self):
        p = CredibilityParams(credibility_weight=1.0, ballast=0.0)
        assert p.credibility_weight == 1.0


# ---------------------------------------------------------------------------
# ExperienceModFactor
# ---------------------------------------------------------------------------


class TestExperienceModFactor:
    @pytest.fixture
    def emod(self):
        params = CredibilityParams(credibility_weight=0.70, ballast=5000.0)
        return ExperienceModFactor(params)

    def test_mod_one_when_actual_equals_expected(self, emod):
        """When actual = expected, mod should be 1.0."""
        mod = emod.predict(expected_losses=20_000, actual_losses=20_000)
        assert mod == pytest.approx(1.0)

    def test_mod_above_one_when_actual_exceeds_expected(self, emod):
        mod = emod.predict(expected_losses=20_000, actual_losses=30_000)
        assert mod > 1.0

    def test_mod_below_one_when_actual_below_expected(self, emod):
        mod = emod.predict(expected_losses=20_000, actual_losses=10_000)
        assert mod < 1.0

    def test_formula_manually(self):
        """Verify formula: Mod = (A*actual + (1-A)*expected + B) / (expected + B)."""
        A, B = 0.60, 3000.0
        expected, actual = 15_000.0, 18_000.0
        expected_mod = (A * actual + (1 - A) * expected + B) / (expected + B)
        params = CredibilityParams(credibility_weight=A, ballast=B)
        emod = ExperienceModFactor(params)
        mod = emod.predict(expected, actual)
        assert mod == pytest.approx(expected_mod)

    def test_zero_credibility_always_returns_one(self):
        """A = 0 means no experience rating; mod = 1.0 regardless of actual losses."""
        params = CredibilityParams(credibility_weight=0.0, ballast=0.0)
        emod = ExperienceModFactor(params)
        mod = emod.predict(expected_losses=20_000, actual_losses=100_000)
        assert mod == pytest.approx(1.0)

    def test_full_credibility_no_ballast(self):
        """A = 1, B = 0: Mod = actual / expected."""
        params = CredibilityParams(credibility_weight=1.0, ballast=0.0)
        emod = ExperienceModFactor(params)
        mod = emod.predict(expected_losses=20_000, actual_losses=30_000)
        assert mod == pytest.approx(30_000 / 20_000)

    def test_cap_applied(self, emod):
        mod = emod.predict(expected_losses=10_000, actual_losses=200_000, cap=2.0)
        assert mod == pytest.approx(2.0)

    def test_floor_applied(self, emod):
        mod = emod.predict(expected_losses=20_000, actual_losses=0, floor=0.5)
        assert mod >= pytest.approx(0.5)

    def test_zero_expected_raises(self, emod):
        with pytest.raises(ValueError, match="expected_losses must be positive"):
            emod.predict(expected_losses=0, actual_losses=1000)

    def test_negative_expected_raises(self, emod):
        with pytest.raises(ValueError, match="expected_losses must be positive"):
            emod.predict(expected_losses=-500, actual_losses=1000)

    def test_predict_batch_returns_dataframe(self, emod):
        df = pl.DataFrame({
            "risk_id": ["A", "B", "C"],
            "expected_losses": [20_000.0, 50_000.0, 10_000.0],
            "actual_losses": [25_000.0, 40_000.0, 8_000.0],
        })
        result = emod.predict_batch(df)
        assert isinstance(result, pl.DataFrame)
        assert "mod_factor" in result.columns
        assert len(result) == 3

    def test_predict_batch_consistent_with_scalar(self, emod):
        """Batch results should match scalar predict() for each row."""
        df = pl.DataFrame({
            "expected_losses": [20_000.0, 50_000.0],
            "actual_losses": [25_000.0, 40_000.0],
        })
        result = emod.predict_batch(df)
        for i in range(len(df)):
            scalar = emod.predict(
                df["expected_losses"][i], df["actual_losses"][i]
            )
            assert result["mod_factor"][i] == pytest.approx(scalar)

    def test_predict_batch_missing_column_raises(self, emod):
        df = pl.DataFrame({"expected_losses": [20_000.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            emod.predict_batch(df)

    def test_predict_batch_cap(self, emod):
        df = pl.DataFrame({
            "expected_losses": [10_000.0],
            "actual_losses": [200_000.0],
        })
        result = emod.predict_batch(df, cap=2.0)
        assert result["mod_factor"][0] == pytest.approx(2.0)

    def test_predict_batch_floor(self, emod):
        df = pl.DataFrame({
            "expected_losses": [20_000.0],
            "actual_losses": [0.0],
        })
        result = emod.predict_batch(df, floor=0.5)
        assert result["mod_factor"][0] >= pytest.approx(0.5)

    def test_sensitivity_returns_dataframe(self, emod):
        df = emod.sensitivity(expected_losses=20_000.0, n_points=20)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 20
        assert "mod_factor" in df.columns
        assert "loss_ratio" in df.columns

    def test_from_exposure_square_root(self):
        """Square root formula: A = sqrt(min(e/e_full, 1))."""
        emod = ExperienceModFactor.from_exposure(
            actual_exposure=250,
            full_credibility_exposure=1000,
            ballast=5000.0,
            credibility_formula="square_root",
        )
        expected_A = np.sqrt(250 / 1000)
        assert emod.params.credibility_weight == pytest.approx(expected_A)

    def test_from_exposure_linear(self):
        """Linear formula: A = min(e/e_full, 1)."""
        emod = ExperienceModFactor.from_exposure(
            actual_exposure=600,
            full_credibility_exposure=1000,
            ballast=2000.0,
            credibility_formula="linear",
        )
        assert emod.params.credibility_weight == pytest.approx(0.60)

    def test_from_exposure_capped_at_one(self):
        """Exposure above full_credibility_exposure should give A = 1.0."""
        emod = ExperienceModFactor.from_exposure(
            actual_exposure=2000,
            full_credibility_exposure=1000,
            ballast=0.0,
        )
        assert emod.params.credibility_weight == pytest.approx(1.0)

    def test_from_exposure_invalid_formula_raises(self):
        with pytest.raises(ValueError, match="Unknown credibility_formula"):
            ExperienceModFactor.from_exposure(1000, 1000, 0.0, "magic")

    def test_ballast_dampens_extreme_losses(self):
        """Higher ballast -> less extreme mod factors."""
        params_low_B = CredibilityParams(credibility_weight=0.70, ballast=100.0)
        params_high_B = CredibilityParams(credibility_weight=0.70, ballast=50_000.0)
        emod_low = ExperienceModFactor(params_low_B)
        emod_high = ExperienceModFactor(params_high_B)

        mod_low = emod_low.predict(expected_losses=10_000, actual_losses=50_000)
        mod_high = emod_high.predict(expected_losses=10_000, actual_losses=50_000)
        # High ballast should produce a smaller (less extreme) mod
        assert abs(mod_high - 1.0) < abs(mod_low - 1.0)


# ---------------------------------------------------------------------------
# ScheduleRating
# ---------------------------------------------------------------------------


class TestScheduleRating:
    @pytest.fixture
    def sr(self):
        rating = ScheduleRating(max_total_debit=0.25, max_total_credit=0.25)
        rating.add_factor("Premises", -0.10, 0.10, "Premises quality")
        rating.add_factor("Management", -0.05, 0.05, "Management quality")
        return rating

    def test_neutral_rating_is_one(self, sr):
        factor = sr.rate({"Premises": 0.0, "Management": 0.0})
        assert factor == pytest.approx(1.0)

    def test_pure_debit(self, sr):
        factor = sr.rate({"Premises": 0.08, "Management": 0.03})
        assert factor == pytest.approx(1.11)

    def test_pure_credit(self, sr):
        factor = sr.rate({"Premises": -0.08, "Management": -0.03})
        assert factor == pytest.approx(0.89)

    def test_mixed_debit_credit(self, sr):
        factor = sr.rate({"Premises": 0.05, "Management": -0.05})
        assert factor == pytest.approx(1.0)

    def test_debit_within_bounds_accepted(self, sr):
        factor = sr.rate({"Premises": 0.10})  # exactly at max
        assert factor == pytest.approx(1.10)

    def test_debit_above_factor_bound_raises(self, sr):
        with pytest.raises(ValueError, match="exceeds max_debit"):
            sr.rate({"Premises": 0.15})

    def test_credit_below_factor_bound_raises(self, sr):
        with pytest.raises(ValueError, match="below min_credit"):
            sr.rate({"Premises": -0.15})

    def test_aggregate_debit_capped(self):
        """Total debit should be capped at max_total_debit."""
        sr = ScheduleRating(max_total_debit=0.15, max_total_credit=0.25)
        sr.add_factor("F1", -0.20, 0.10)
        sr.add_factor("F2", -0.20, 0.10)
        # 0.10 + 0.10 = 0.20 debit, but cap is 0.15
        factor = sr.rate({"F1": 0.10, "F2": 0.10})
        assert factor == pytest.approx(1.15)

    def test_aggregate_credit_capped(self):
        """Total credit should be capped at max_total_credit."""
        sr = ScheduleRating(max_total_debit=0.25, max_total_credit=0.15)
        sr.add_factor("F1", -0.10, 0.20)
        sr.add_factor("F2", -0.10, 0.20)
        # -0.10 + -0.10 = -0.20 credit, but cap is 0.15
        factor = sr.rate({"F1": -0.10, "F2": -0.10})
        assert factor == pytest.approx(0.85)

    def test_unknown_factor_raises(self, sr):
        with pytest.raises(ValueError, match="Unknown factor"):
            sr.rate({"NonExistent": 0.05})

    def test_duplicate_factor_raises(self, sr):
        with pytest.raises(ValueError, match="already exists"):
            sr.add_factor("Premises", -0.05, 0.05)

    def test_invalid_factor_bounds_raises(self):
        sr = ScheduleRating()
        with pytest.raises(ValueError, match="min_credit must be <= 0"):
            sr.add_factor("Bad", min_credit=0.05, max_debit=0.10)

    def test_invalid_max_debit_raises(self):
        sr = ScheduleRating()
        with pytest.raises(ValueError, match="max_debit must be >= 0"):
            sr.add_factor("Bad", min_credit=-0.05, max_debit=-0.10)

    def test_negative_max_total_debit_raises(self):
        with pytest.raises(ValueError, match="max_total_debit"):
            ScheduleRating(max_total_debit=-0.10)

    def test_negative_max_total_credit_raises(self):
        with pytest.raises(ValueError, match="max_total_credit"):
            ScheduleRating(max_total_credit=-0.10)

    def test_chaining_add_factor(self):
        sr = (
            ScheduleRating()
            .add_factor("A", -0.10, 0.10)
            .add_factor("B", -0.05, 0.05)
        )
        assert len(sr.factor_names) == 2

    def test_summary_returns_dataframe(self, sr):
        df = sr.summary()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns

    def test_empty_summary_returns_empty_dataframe(self):
        sr = ScheduleRating()
        df = sr.summary()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_rate_batch_returns_dataframe(self, sr):
        df = pl.DataFrame({
            "risk_id": ["R1", "R2", "R3"],
            "Premises": [0.05, -0.05, 0.00],
            "Management": [0.03, -0.02, 0.01],
        })
        result = sr.rate_batch(df)
        assert isinstance(result, pl.DataFrame)
        assert "schedule_factor" in result.columns
        assert len(result) == 3

    def test_rate_batch_consistent_with_scalar(self, sr):
        df = pl.DataFrame({
            "Premises": [0.05, -0.05],
            "Management": [0.03, -0.02],
        })
        result = sr.rate_batch(df)
        expected_0 = sr.rate({"Premises": 0.05, "Management": 0.03})
        expected_1 = sr.rate({"Premises": -0.05, "Management": -0.02})
        assert result["schedule_factor"][0] == pytest.approx(expected_0)
        assert result["schedule_factor"][1] == pytest.approx(expected_1)

    def test_rate_batch_no_factor_columns_raises(self, sr):
        df = pl.DataFrame({"unrelated": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No recognised factor columns"):
            sr.rate_batch(df)

    def test_rate_batch_out_of_range_raises(self, sr):
        df = pl.DataFrame({
            "Premises": [0.50],  # exceeds max_debit of 0.10
            "Management": [0.00],
        })
        with pytest.raises(ValueError, match="out-of-range"):
            sr.rate_batch(df)

    def test_repr(self, sr):
        r = repr(sr)
        assert "ScheduleRating" in r
        assert "n_factors=2" in r
