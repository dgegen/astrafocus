import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Longitude


def angle_difference_deg(a, b):
    """
    a = angle_difference_deg(Longitude(359.0 * u.deg),Longitude(1.0 * u.deg))
    b = angle_difference_deg(Longitude(1.0 * u.deg),Longitude(359.0 * u.deg))

    angle_difference_deg(df.ra.to_numpy() * u.deg, zenith_neighbourhood.zenith.ra)
    """
    if not (isinstance(a, (u.Quantity, Angle)) and isinstance(a, (u.Quantity, Angle))):
        diff = Longitude(a * u.deg) - Longitude(b * u.deg)
    else:
        diff = Longitude(a) - Longitude(b)

    return np.minimum(Longitude(diff), Longitude(-diff))
