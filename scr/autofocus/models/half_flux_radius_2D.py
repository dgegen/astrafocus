import numpy as np
import scipy

from astropy.modeling.models import Disk2D


class HalfFluxRadius2D(Disk2D):
    """
    hfr_2D = HalfFluxRadius2D()
    star_data=np.random.uniform(size=(20, 20))
    x, y = np.indices(star_data.shape)
    hfr_2D.fit(star_data, x, y, scale_factor=5)
    """

    def __init__(self, amplitude=1, x_0=0, y_0=0, R_0=1, scale_factor=1, **kwargs):
        super().__init__(amplitude=amplitude, x_0=x_0, y_0=y_0, R_0=R_0, **kwargs)

    def fit(self, x, y, star_data, *args, **kwargs):
        """
        Source
        https://en.wikipedia.org/wiki/Half_flux_diameter#cite_note-4
        https://www.lost-infinity.com/the-half-flux-diameter-hfd-for-a-perfectly-normal-distributed-star/
        """
        # Resample the image to a higher resolution using interpolation
        centroid = np.array([np.sum(x * star_data), np.sum(y * star_data)]) / np.sum(star_data)

        # Calculate the radius of each pixel from the centroid
        distances = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)

        # Estimate the half flux radius (HFR)
        hfr = np.sum(star_data * distances) / np.sum(star_data)

        # Calculate the Half Flux Diameter (HFD)
        hfd_high_res = 2 * hfr

        # Convert back to the original image size
        hfd = hfd_high_res

        self.x_0, self.y_0 = centroid
        self.R_0 = hfd_high_res

        return self

    @staticmethod
    def optimize_hfr(high_res_image, distances, hfr):
        def hfr_function(hfr_tmp):
            return np.abs(
                np.sum(high_res_image[distances > hfr_tmp])
                - np.sum(high_res_image[distances <= hfr_tmp])
            )

        def hfr_minimise(hfr_tmp):
            return hfr_function(hfr_tmp) / hfr_function(hfr)

        res = scipy.optimize.minimize(
            hfr_minimise, x0=hfr, options={"eps": 10}, bounds=scipy.optimize.Bounds(0, 2 * hfr)
        )
        hfr = res.x
        return hfr
