from abc import ABC, abstractmethod
from astropy.coordinates import SkyCoord


class TelescopeInterface(ABC):
    """
    A calss to interface the pointing of the telescope to a specific coordinate
    in the equatorial coordinate system.
    """

    def point_to(self, coordinates):
        """
        Point the telescope to a specific coordinate in the equatorial coordinate system.

        Parameters
        ----------
        coordinates : `~astropy.coordinates.SkyCoord`
            The ICRS coordinates that should be in the centre of the CCD.
        """
        self.validate_arguments(coordinates)
        self.set_telescope_position(coordinates)

    @abstractmethod
    def set_telescope_position(self, coordinates: SkyCoord):
        """
        Point the telescope to a specific coordinate in the equatorial coordinate system.

        Parameters
        ----------
        coordinates : `~astropy.coordinates.SkyCoord`
            The ICRS coordinates that should be in the centre of the CCD.
        """
        pass

    def validate_arguments(self, coordinates: SkyCoord):
        if not isinstance(coordinates, SkyCoord):
            raise ValueError("Coordinates must be an instance of astropy.coordinates.SkyCoord")

    def __repr__(self) -> str:
        return f"TelescopeInterface()"


class TrivialTelescope(TelescopeInterface):
    """
    Trivial implementation to set the telescope position for testing purposes.
    """
    @staticmethod
    def set_telescope_position(coordinates: SkyCoord):
        pass
