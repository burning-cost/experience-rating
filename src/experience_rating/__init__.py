"""
experience-rating: NCD/bonus-malus systems, experience modification factors,
and schedule rating for UK non-life insurance pricing.

Modules:
    bonus_malus: BonusMalusScale, BonusMalusSimulator, ClaimThreshold
    experience_mod: ExperienceModFactor, ScheduleRating
"""

from experience_rating.bonus_malus import (
    BonusMalusLevel,
    BonusMalusScale,
    BonusMalusSimulator,
    ClaimThreshold,
)
from experience_rating.experience_mod import (
    ExperienceModFactor,
    ScheduleRating,
)

__all__ = [
    "BonusMalusLevel",
    "BonusMalusScale",
    "BonusMalusSimulator",
    "ClaimThreshold",
    "ExperienceModFactor",
    "ScheduleRating",
]

__version__ = "0.1.0"
