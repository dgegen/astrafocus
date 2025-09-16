from astrafocus.autofocuser import AnalyticResponseAutofocuser, NonParametricResponseAutofocuser
from astrafocus.extremum_estimators import ExtremumEstimatorRegistry
from astrafocus.focus_measure_operators import (
    AbsoluteGradientFocusMeasure,
    AutoCorrelationFocusMeasure,
    BrennerFocusMeasure,
    FFTFocusMeasureTan2022,
    FFTPhaseMagnitudeProductFocusMeasure,
    FFTPowerFocusMeasure,
    LaplacianFocusMeasure,
    NormalizedVarianceFocusMeasure,
    SquaredGradientFocusMeasure,
    TenengradFocusMeasure,
    VarianceOfLaplacianFocusMeasure,
)
from astrafocus.star_size_focus_measure_operators import (
    GaussianStarFocusMeasure,
    HFRStarFocusMeasure,
)
from astrafocus.utils.logger import get_logger

logger = get_logger()

__all__ = [
    "AnalyticResponseAutofocuser",
    "NonParametricResponseAutofocuser",
    "FocusMeasureOperatorRegistry",
    "ExtremumEstimatorRegistry",
]


class FocusMeasureOperatorRegistry:
    """Registry mapping string keys to focus measure operator classes.

    Examples
    --------
    >>> from astrafocus import FocusMeasureOperatorRegistry
    >>> FocusMeasureOperatorRegistry.list()
    >>> FocusMeasureOperatorRegistry.from_name("fft")
    """

    _operators = {
        "hfr": HFRStarFocusMeasure,
        "gauss": GaussianStarFocusMeasure,
        "fft": FFTFocusMeasureTan2022,
        "fft_power": FFTPowerFocusMeasure,
        "fft_phase_magnitude_product": FFTPhaseMagnitudeProductFocusMeasure,
        "normalized_variance": NormalizedVarianceFocusMeasure,
        "brenner": BrennerFocusMeasure,
        "tenengrad": TenengradFocusMeasure,
        "laplacian": LaplacianFocusMeasure,
        "variance_laplacian": VarianceOfLaplacianFocusMeasure,
        "absolute_gradient": AbsoluteGradientFocusMeasure,
        "squared_gradient": SquaredGradientFocusMeasure,
        "auto_correlation": AutoCorrelationFocusMeasure,
    }

    @classmethod
    def get(cls, key: str, default=HFRStarFocusMeasure):
        """Get a focus measure operator class by fuzzy matching the key. Returns hfr if not found."""
        return cls._operators.get(key, default)

    @classmethod
    def from_name(cls, key: str):
        """Get a focus measure operator class by fuzzy matching the key. Returns hfr if not found."""
        key = key.lower().replace("-", "_").replace(" ", "_")
        # Exact match
        if key in cls._operators:
            return cls._operators[key]
        # Fuzzy match
        words = key.split("_")
        for name, operator in cls._operators.items():
            if any(word in name for word in words):
                return operator
        logger.warning(f"Focus measure operator '{key}' not found. Using 'hfr' instead.")
        return cls._operators["hfr"]

    @classmethod
    def list(cls):
        """List all available focus measure operators."""
        return
