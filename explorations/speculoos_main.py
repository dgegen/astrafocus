from typing import Optional, Tuple

import astropy
import numpy as np
from autofocus.autofocuser import AnalyticResponseAutofocuser
from autofocus.interface.focuser import FocuserInterface
from autofocus.interface.pointer import PointerInterface
from autofocus.interface.telescope import TelescopeInterface
from autofocus.interface.telescope_specs import TelescopeSpecs
from autofocus.star_size_focus_measure_operators import HFRStarFocusMeasure
from autofocus.targeting.zenith_neighbourhood_query import ZenithNeighbourhoodQuery

TELESCOPE_SPECS = TelescopeSpecs.load_telescope_config(file_path="file_path/to/speculoos.yaml")


class TelescopePointer(PointerInterface):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def set_telescope_position(coordinates: astropy.coordinates.SkyCoord):
        """
        Calling this function should take as long as it takes to move the telescope to the desired position.
        """
        # TODOD
        pass


class TelescopeFocuser(FocuserInterface):
    def __init__(self, allowed_range: Tuple[int, int] = TELESCOPE_SPECS.focus_range):
        current_position = self.get_current_position()
        super().__init__(current_position=current_position, allowed_range=allowed_range)

    def move_focuser_to_position(self, new_position: int):
        """
        Calling this function should take as long as it takes to move the focuser to the desired position.
        """
        # TODOD
        pass

    def get_current_position(self):
        # TODOD
        pass


class Telescope(TelescopeInterface):
    def __init__(self, focuser: TelescopeFocuser, pointer: TelescopePointer):
        super().__init__(focuser=focuser, pointer=pointer)

    def take_observation(self, texp: float):
        """
        Calling this function should take as long as it takes to make the observation.
        """
        # TODOD
        pass


def focus_speculoos(
    observation_time: astropy.time.Time = None,
    maximal_zenith_angle: astropy.coordinates.Angle = None,
    g_mag_range: Optional[Tuple[float, float]] = None,
    j_mag_range: Optional[Tuple[float, float]] = None,
):
    """
    # Should we consider adding 30 seconds of buffer into the future
    # to compensate the expected query time?
    observation_time = None
    maximal_zenith_angle = 10 * astropy.units.deg
    g_mag_range = None
    j_mag_range = None
    """

    # Pointing the telescope
    zenith_neighbourhood_query = ZenithNeighbourhoodQuery.from_telescope_specs(
        telescope_specs=TELESCOPE_SPECS,
        observation_time=observation_time,
        maximal_zenith_angle=maximal_zenith_angle,
    )
    znqr_full = zenith_neighbourhood_query.query_shardwise(n_sub_div=20)

    # Mask by magnitude
    znqr = znqr_full.mask_by_magnitude(
        g_mag_range=g_mag_range or TELESCOPE_SPECS.g_mag_range,
        j_mag_range=j_mag_range or TELESCOPE_SPECS.j_mag_range,
    )

    # Determine the number of stars that would be on the ccd
    # if the telescope was centred on a given star
    znqr.determine_stars_in_neighbourhood()

    # Find star closest to zenith with desired magnitude that is alone in a nbh of min_deg degrees
    znqr.sort_values(["zenith_angle", "n"], ascending=[True, True])

    centre_coordinates = znqr.get_sky_coord_of_select_star(np.argmax(znqr.n == 1))

    # Point telescope
    telescope = Telescope(focuser=TelescopeFocuser(), pointer=TelescopePointer())
    telescope.pointer.set_telescope_position(centre_coordinates)

    # Start autofocus, e.g.
    ARAF = AnalyticResponseAutofocuser(
        telescope_interface=telescope,
        focus_measure_operator=HFRStarFocusMeasure,
        n_steps=(30, 20),
        n_exposures=(1, 1),
        decrease_search_range=True,
        percent_to_cut=60,
    )

    ARAF.run()
