from enum import Enum

from astrafocus.autofocuser import AnalyticResponseAutofocuser, NonParametricResponseAutofocuser
from astrafocus.extremum_estimators import ExtremumEstimators
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
    "FocusMeasureOperators",
    "ExtremumEstimators",
]


class FocusMeasureOperators(Enum):
    """Enum mapping string keys to focus measure operator classes.

    Examples
    --------
    >>> from astrafocus import FocusMeasureOperators
    >>> FocusMeasureOperators.list()
    >>> FocusMeasureOperators.hdr
    >>> FocusMeasureOperators.from_name("fft")  # Get the class by fuzzy matching
    """

    hfr = HFRStarFocusMeasure
    gauss = GaussianStarFocusMeasure
    fft = FFTFocusMeasureTan2022
    fft_power = FFTPowerFocusMeasure
    fft_phase_magnitude_product = FFTPhaseMagnitudeProductFocusMeasure
    normalized_variance = NormalizedVarianceFocusMeasure
    brenner = BrennerFocusMeasure
    tenengrad = TenengradFocusMeasure
    laplacian = LaplacianFocusMeasure
    variance_laplacian = VarianceOfLaplacianFocusMeasure
    absolute_gradient = AbsoluteGradientFocusMeasure
    squared_gradient = SquaredGradientFocusMeasure
    auto_correlation = AutoCorrelationFocusMeasure

    @classmethod
    def from_name(cls, key: str):
        """Get a FocusMeasureOperatorType by fuzzy matching the key. Returns hfr if not found."""
        key = key.lower()
        for member in cls:
            if key in member.name.lower():
                return member.value

        logger.warning(f"Focus measure operator '{key}' not found. Using 'hfr' instead.")
        return cls.hfr.value

    @classmethod
    def list(cls):
        return [member.name for member in cls]
