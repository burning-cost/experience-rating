"""
experience-rating: NCD/bonus-malus systems, experience modification factors,
and schedule rating for UK non-life insurance pricing.

Modules:
    bonus_malus: BonusMalusLevel, BonusMalusScale, BonusMalusSimulator,
                 ClaimThreshold, TransitionRules
    experience_mod: CredibilityParams, ExperienceModFactor, ScheduleRating
"""

from experience_rating.bonus_malus import (
    BonusMalusLevel,
    BonusMalusScale,
    BonusMalusSimulator,
    ClaimThreshold,
    TransitionRules,
)
from experience_rating.experience_mod import (
    CredibilityParams,
    ExperienceModFactor,
    ScheduleRating,
)

__all__ = [
    "BonusMalusLevel",
    "BonusMalusScale",
    "BonusMalusSimulator",
    "ClaimThreshold",
    "CredibilityParams",
    "ExperienceModFactor",
    "ScheduleRating",
    "TransitionRules",
]

__version__ = "0.1.2"
