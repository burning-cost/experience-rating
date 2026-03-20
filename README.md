# experience-rating — Deprecated

This package has been superseded by [insurance-credibility](https://github.com/burning-cost/insurance-credibility).

All functionality — `BonusMalusSimulator`, `BonusMalusScale`, `BonusMalusLevel`, `ExperienceModFactor`, `ScheduleRating`, and `CredibilityParams` — is now part of insurance-credibility under the `insurance_credibility.experience` subpackage, alongside more advanced Bayesian experience rating models.

## Migration

```bash
pip install insurance-credibility
```

```python
# Before
from experience_rating import BonusMalusSimulator, ExperienceModFactor

# After
from insurance_credibility.experience import BonusMalusSimulator, ExperienceModFactor
```

This repository is archived and will not receive further updates.
