"""
Tests for BonusMalusScale, BonusMalusSimulator, and ClaimThreshold.
"""

import numpy as np
import polars as pl
import pytest

from experience_rating.bonus_malus import (
    BonusMalusLevel,
    BonusMalusScale,
    BonusMalusSimulator,
    ClaimThreshold,
    TransitionRules,
)


# ---------------------------------------------------------------------------
# TransitionRules
# ---------------------------------------------------------------------------


class TestTransitionRules:
    def test_claim_free(self):
        tr = TransitionRules(claim_free_level=5, claim_levels={1: 3, 2: 0})
        assert tr.next_level(0) == 5

    def test_one_claim(self):
        tr = TransitionRules(claim_free_level=5, claim_levels={1: 3, 2: 0})
        assert tr.next_level(1) == 3

    def test_two_claims(self):
        tr = TransitionRules(claim_free_level=5, claim_levels={1: 3, 2: 0})
        assert tr.next_level(2) == 0

    def test_excess_claims_uses_minimum(self):
        """More claims than explicit rules should use the worst (lowest) destination."""
        tr = TransitionRules(claim_free_level=5, claim_levels={1: 3, 2: 0})
        # 3 claims has no explicit rule; should fall back to min(values) = 0
        assert tr.next_level(3) == 0

    def test_excess_claims_with_asymmetric_rules(self):
        """Test fallback when claim_levels has non-zero minimum."""
        tr = TransitionRules(claim_free_level=7, claim_levels={1: 4, 2: 2})
        # 5 claims -> min(4, 2) = 2
        assert tr.next_level(5) == 2


# ---------------------------------------------------------------------------
# BonusMalusScale construction
# ---------------------------------------------------------------------------


class TestBonusMalusScaleConstruction:
    def test_uk_standard_has_10_levels(self):
        scale = BonusMalusScale.from_uk_standard()
        assert len(scale) == 10

    def test_uk_standard_level_0_is_no_ncd(self):
        scale = BonusMalusScale.from_uk_standard()
        assert scale.levels[0].ncd_percent == 0
        assert scale.levels[0].premium_factor == pytest.approx(1.0)

    def test_uk_standard_level_9_is_max_ncd(self):
        scale = BonusMalusScale.from_uk_standard()
        assert scale.levels[9].ncd_percent == 65
        assert scale.levels[9].premium_factor == pytest.approx(0.35)

    def test_uk_standard_premium_factors_decreasing(self):
        """Higher levels should have lower (better) premium factors."""
        scale = BonusMalusScale.from_uk_standard()
        factors = [lvl.premium_factor for lvl in scale.levels]
        assert all(factors[i] >= factors[i + 1] for i in range(len(factors) - 1))

    def test_uk_standard_claim_free_moves_up(self):
        """Claim-free transition from level i should go to level i+1 (capped at max)."""
        scale = BonusMalusScale.from_uk_standard()
        for i in range(len(scale) - 1):
            assert scale.levels[i].transitions.claim_free_level == i + 1
        # Max level stays at max
        assert scale.levels[9].transitions.claim_free_level == 9

    def test_uk_standard_one_claim_moves_back_two(self):
        """One claim should set level to max(current - 2, 0)."""
        scale = BonusMalusScale.from_uk_standard()
        for i in range(len(scale)):
            expected_dest = max(i - 2, 0)
            actual_dest = scale.levels[i].transitions.claim_levels[1]
            assert actual_dest == expected_dest, f"Level {i}: expected {expected_dest}, got {actual_dest}"

    def test_uk_standard_two_claims_resets_to_zero(self):
        """Two or more claims should reset to level 0."""
        scale = BonusMalusScale.from_uk_standard()
        for lvl in scale.levels:
            assert lvl.transitions.claim_levels[2] == 0

    def test_from_dict_roundtrip(self):
        """from_dict should produce an equivalent scale."""
        original = BonusMalusScale.from_uk_standard()
        spec = {
            "levels": [
                {
                    "index": lvl.index,
                    "name": lvl.name,
                    "premium_factor": lvl.premium_factor,
                    "ncd_percent": lvl.ncd_percent,
                    "transitions": {
                        "claim_free_level": lvl.transitions.claim_free_level,
                        "claim_levels": {
                            str(k): v for k, v in lvl.transitions.claim_levels.items()
                        },
                    },
                }
                for lvl in original.levels
            ]
        }
        reconstructed = BonusMalusScale.from_dict(spec)
        assert len(reconstructed) == len(original)
        for i in range(len(original)):
            assert reconstructed.levels[i].premium_factor == pytest.approx(
                original.levels[i].premium_factor
            )

    def test_custom_scale_from_dict(self):
        """Build a minimal 3-level custom scale."""
        spec = {
            "levels": [
                {
                    "index": 0,
                    "name": "Basic",
                    "premium_factor": 1.20,
                    "ncd_percent": None,
                    "transitions": {
                        "claim_free_level": 1,
                        "claim_levels": {"1": 0},
                    },
                },
                {
                    "index": 1,
                    "name": "Standard",
                    "premium_factor": 1.00,
                    "ncd_percent": None,
                    "transitions": {
                        "claim_free_level": 2,
                        "claim_levels": {"1": 0},
                    },
                },
                {
                    "index": 2,
                    "name": "Preferred",
                    "premium_factor": 0.80,
                    "ncd_percent": None,
                    "transitions": {
                        "claim_free_level": 2,
                        "claim_levels": {"1": 0},
                    },
                },
            ]
        }
        scale = BonusMalusScale.from_dict(spec)
        assert len(scale) == 3
        assert scale.levels[2].premium_factor == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# BonusMalusScale validation
# ---------------------------------------------------------------------------


class TestBonusMalusScaleValidation:
    def test_empty_scale_raises(self):
        with pytest.raises(ValueError, match="at least one level"):
            BonusMalusScale([])

    def test_gap_in_indices_raises(self):
        tr = TransitionRules(claim_free_level=1, claim_levels={1: 0})
        levels = [
            BonusMalusLevel(0, "L0", 1.0, tr),
            BonusMalusLevel(2, "L2", 0.8, tr),  # index 1 is missing
        ]
        with pytest.raises(ValueError, match="indices must be"):
            BonusMalusScale(levels)

    def test_invalid_transition_destination_raises(self):
        tr_bad = TransitionRules(claim_free_level=5, claim_levels={1: 0})  # dest 5 invalid
        levels = [
            BonusMalusLevel(0, "L0", 1.0, tr_bad),
            BonusMalusLevel(1, "L1", 0.8, TransitionRules(0, {1: 0})),
        ]
        with pytest.raises(ValueError, match="not a valid index"):
            BonusMalusScale(levels)

    def test_negative_premium_factor_raises(self):
        tr = TransitionRules(claim_free_level=1, claim_levels={1: 0})
        levels = [
            BonusMalusLevel(0, "L0", -0.5, tr),
            BonusMalusLevel(1, "L1", 0.8, TransitionRules(1, {1: 0})),
        ]
        with pytest.raises(ValueError, match="premium_factor must be positive"):
            BonusMalusScale(levels)


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_matrix_shape(self):
        scale = BonusMalusScale.from_uk_standard()
        T = scale.transition_matrix(claim_frequency=0.10)
        assert T.shape == (10, 10)

    def test_matrix_row_stochastic(self):
        """Each row must sum to 1."""
        scale = BonusMalusScale.from_uk_standard()
        T = scale.transition_matrix(claim_frequency=0.10)
        row_sums = T.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_matrix_nonnegative(self):
        scale = BonusMalusScale.from_uk_standard()
        T = scale.transition_matrix(claim_frequency=0.10)
        assert (T >= 0).all()

    def test_zero_frequency_all_move_up(self):
        """At zero claim frequency, all policyholders move up one level."""
        scale = BonusMalusScale.from_uk_standard()
        T = scale.transition_matrix(claim_frequency=1e-12)
        # Level 0 should transition almost certainly to level 1
        assert T[0, 1] > 0.999
        # Level 9 stays at 9
        assert T[9, 9] > 0.999

    def test_high_frequency_increases_lower_level_prob(self):
        """At high claim frequency, level 0 destination probabilities increase."""
        scale = BonusMalusScale.from_uk_standard()
        T_low = scale.transition_matrix(claim_frequency=0.05)
        T_high = scale.transition_matrix(claim_frequency=0.50)
        # From level 9, probability of ending at level 0 should be higher at high freq
        assert T_high[9, 0] > T_low[9, 0]

    def test_summary_returns_dataframe(self):
        scale = BonusMalusScale.from_uk_standard()
        df = scale.summary()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert "premium_factor" in df.columns


# ---------------------------------------------------------------------------
# BonusMalusSimulator
# ---------------------------------------------------------------------------


class TestBonusMalusSimulator:
    @pytest.fixture
    def sim(self):
        scale = BonusMalusScale.from_uk_standard()
        return BonusMalusSimulator(scale, claim_frequency=0.10, rng_seed=42)

    def test_simulate_returns_dataframe(self, sim):
        df = sim.simulate(n_policyholders=1000, n_years=5)
        assert isinstance(df, pl.DataFrame)

    def test_simulate_year_count(self, sim):
        """Output should have years 0 through n_years."""
        df = sim.simulate(n_policyholders=1000, n_years=5)
        years = sorted(df["year"].unique().to_list())
        assert years == list(range(6))  # 0, 1, 2, 3, 4, 5

    def test_simulate_proportions_sum_to_one(self, sim):
        """Proportions in each year should sum to 1.0."""
        df = sim.simulate(n_policyholders=1000, n_years=5)
        year_sums = df.group_by("year").agg(pl.sum("proportion")).sort("year")
        np.testing.assert_allclose(
            year_sums["proportion"].to_list(),
            [1.0] * 6,
            atol=1e-6,
        )

    def test_simulate_starts_at_level_zero(self, sim):
        """Year 0 should have all policyholders at level 0."""
        df = sim.simulate(n_policyholders=1000, n_years=5)
        year0 = df.filter(pl.col("year") == 0)
        level0_prop = year0.filter(pl.col("level") == 0)["proportion"][0]
        assert level0_prop == pytest.approx(1.0)

    def test_simulate_custom_starting_level(self, sim):
        """Starting at level 5 should have all policyholders there at year 0."""
        df = sim.simulate(n_policyholders=1000, n_years=5, starting_level=5)
        year0 = df.filter(pl.col("year") == 0)
        level5_prop = year0.filter(pl.col("level") == 5)["proportion"][0]
        assert level5_prop == pytest.approx(1.0)

    def test_simulate_invalid_starting_level(self, sim):
        with pytest.raises(ValueError, match="starting_level"):
            sim.simulate(n_policyholders=100, n_years=5, starting_level=99)

    def test_stationary_analytical_sums_to_one(self, sim):
        dist = sim.stationary_distribution(method="analytical")
        total = dist["stationary_prob"].sum()
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_stationary_analytical_nonnegative(self, sim):
        dist = sim.stationary_distribution(method="analytical")
        assert (dist["stationary_prob"] >= 0).all()

    def test_stationary_simulation_sums_to_one(self, sim):
        dist = sim.stationary_distribution(method="simulation")
        total = dist["stationary_prob"].sum()
        assert total == pytest.approx(1.0, abs=0.01)  # looser tolerance for sim

    def test_stationary_analytical_vs_simulation(self, sim):
        """Analytical and simulation distributions should agree within ~3%."""
        dist_a = sim.stationary_distribution(method="analytical")
        dist_s = sim.stationary_distribution(method="simulation")
        probs_a = dist_a.sort("level")["stationary_prob"].to_numpy()
        probs_s = dist_s.sort("level")["stationary_prob"].to_numpy()
        np.testing.assert_allclose(probs_a, probs_s, atol=0.03)

    def test_expected_premium_factor_is_between_zero_and_one(self, sim):
        """For standard NCD scale, EPF should be below 1.0 (net benefit from NCD)."""
        epf = sim.expected_premium_factor(method="analytical")
        assert 0.0 < epf < 1.0

    def test_higher_frequency_higher_expected_premium(self):
        """Higher claim frequency -> more time at lower NCD -> higher EPF."""
        scale = BonusMalusScale.from_uk_standard()
        sim_low = BonusMalusSimulator(scale, claim_frequency=0.05)
        sim_high = BonusMalusSimulator(scale, claim_frequency=0.30)
        epf_low = sim_low.expected_premium_factor()
        epf_high = sim_high.expected_premium_factor()
        assert epf_high > epf_low

    def test_invalid_method_raises(self, sim):
        with pytest.raises(ValueError, match="Unknown method"):
            sim.stationary_distribution(method="monte_carlo_special")


# ---------------------------------------------------------------------------
# ClaimThreshold
# ---------------------------------------------------------------------------


class TestClaimThreshold:
    @pytest.fixture
    def ct(self):
        scale = BonusMalusScale.from_uk_standard()
        return ClaimThreshold(scale, discount_rate=0.05)

    def test_threshold_at_level_0_is_zero(self, ct):
        """At level 0, claiming has no NCD cost (already at bottom)."""
        # At level 0, one claim still goes to max(0-2, 0) = 0. So no NCD cost.
        th = ct.threshold(current_level=0, annual_premium=500.0)
        assert th == pytest.approx(0.0, abs=1e-6)

    def test_threshold_at_high_level_is_positive(self, ct):
        """At high NCD, claiming has a real financial cost."""
        th = ct.threshold(current_level=9, annual_premium=800.0)
        assert th > 0

    def test_threshold_increases_with_level(self, ct):
        """Higher NCD level -> more to lose -> higher threshold."""
        annual_premium = 700.0
        thresholds = [
            ct.threshold(lvl, annual_premium) for lvl in range(10)
        ]
        # Not strictly monotone (level 0 and 1 both fall to 0 after 1 claim)
        # But threshold at level 9 should exceed threshold at level 3
        assert thresholds[9] > thresholds[3]

    def test_threshold_increases_with_horizon(self, ct):
        """Longer horizon -> more future NCD cost -> higher threshold."""
        th_short = ct.threshold(current_level=8, annual_premium=800.0, years_horizon=1)
        th_long = ct.threshold(current_level=8, annual_premium=800.0, years_horizon=5)
        assert th_long > th_short

    def test_should_claim_large_loss(self, ct):
        """A very large loss should always be worth claiming."""
        assert ct.should_claim(
            current_level=9, claim_amount=50_000, annual_premium=800.0
        )

    def test_should_not_claim_tiny_loss_at_high_ncd(self, ct):
        """A trivial loss at high NCD level should not be claimed."""
        assert not ct.should_claim(
            current_level=9, claim_amount=10.0, annual_premium=800.0
        )

    def test_threshold_curve_returns_dataframe(self, ct):
        df = ct.threshold_curve(current_level=7, annual_premium=700.0, max_horizon=5)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert "threshold_amount" in df.columns

    def test_threshold_curve_is_increasing(self, ct):
        """Threshold should increase (or stay flat) as horizon grows."""
        df = ct.threshold_curve(current_level=7, annual_premium=700.0, max_horizon=8)
        thresholds = df.sort("years_horizon")["threshold_amount"].to_list()
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1] + 1e-6

    def test_full_analysis_has_all_levels(self, ct):
        df = ct.full_analysis(annual_premium=800.0)
        assert len(df) == 10
        assert "claiming_threshold" in df.columns

    def test_invalid_level_raises(self, ct):
        with pytest.raises(ValueError, match="out of range"):
            ct.threshold(current_level=99, annual_premium=500.0)

    def test_higher_premium_gives_higher_threshold(self, ct):
        """More expensive policy -> NCD worth more -> higher claiming threshold."""
        th_low = ct.threshold(current_level=8, annual_premium=400.0)
        th_high = ct.threshold(current_level=8, annual_premium=1200.0)
        assert th_high > th_low
