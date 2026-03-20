import warnings

warnings.warn(
    "experience-rating is deprecated. Use insurance-credibility instead:\n"
    "  pip install insurance-credibility\n"
    "  from insurance_credibility.experience import StaticCredibilityModel, ClaimsHistory\n"
    "  from insurance_credibility.experience import DynamicPoissonGammaModel, SurrogateModel\n"
    "This package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location for backwards compatibility
from insurance_credibility.experience import *  # noqa: F401,F403
