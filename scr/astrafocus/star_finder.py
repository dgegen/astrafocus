from typing import Optional

import astropy
import numpy as np
import photutils
from photutils.detection import DAOStarFinder

from astrafocus.utils.typing import ImageType

class StarFinder:
    """
    Examples
    -------
    TargetFinder(ref_image)
    """

    def __init__(
        self,
        ref_image: ImageType,
        fwhm: float = 2.0,
        star_find_threshold: float = 5.0,
        absolute_detection_limit: float = 0.0,
        saturation_threshold: Optional[float] = None,
    ) -> None:
        self.ref_image = ref_image
        self.fwhm = fwhm
        self.star_find_threshold = star_find_threshold
        self.absolute_detection_limit = absolute_detection_limit
        self.saturation_threshold = saturation_threshold or np.inf
        # - not oversaturated

        mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
        self.ref_background = median
        self.ref_std = std

        potential_targets = self._find_sources()
        self.selected_stars = self.select_target_stars(potential_targets)

    def select_target_stars(self, potential_targets):
        # This could be achieved with peakmax
        selected_stars = potential_targets[potential_targets["peak"] <= self.saturation_threshold]

        return selected_stars

    def _find_sources(self):
        return self.find_sources(
            self.ref_image,
            fwhm=self.fwhm,
            threshold=np.maximum(self.absolute_detection_limit, self.star_find_threshold),
            std=self.ref_std,
            background=self.ref_background,
        )

    @staticmethod
    def find_sources(
        ref_image: ImageType,
        fwhm: float = 1.0,
        threshold: float = 5.0,
        std=None,
        background=None,
        peakmax=None,
    ):
        """Detect and locate stars in the reference image"""
        if std is None or background is None:
            # Calculate summary statistics without pixels that are above or below a sigma from the median
            mean, median, std = astropy.stats.sigma_clipped_stats(ref_image, sigma=3.0)
            background = median

        daofind = DAOStarFinder(fwhm=fwhm, threshold=std * threshold, brightest=None, peakmax=None)
        sources = daofind(ref_image - background)
        sources.sort("flux", reverse=True)
        return sources
