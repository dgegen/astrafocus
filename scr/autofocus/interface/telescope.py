from abc import ABC, abstractmethod

from autofocus.interface.focuser import FocuserInterface
from autofocus.interface.pointer import PointerInterface
from autofocus.utils.typing import ImageType


class TelescopeInterface(ABC):
    """
    Abstract base class representing a telescope interface.
    
    Parameters
    ----------
    focuser : FocuserInterface
        The telescope focuser.
    pointer : TelescopePointer
        The telescope pointer.

    Examples
    --------
    >>> from autofocus.interface.focuser import TrivialFocuser
    >>> from autofocus.interface.pointer import TrivialPointer
    >>> telescope_focuser = TrivialFocuser(current_position=0, allowed_range=(0, 1000))
    >>> telescope_pointer = TrivialPointer()
    >>> telescope_pointer._point_to_alt_az = lambda altitude, azimuth: None
    >>> telescope_interface = TelescopeInterface(
        focuser=telescope_focuser, pointer=telescope_pointer
    )
    """
    def __init__(self, focuser: FocuserInterface, pointer: PointerInterface):
        """
        Initialize the TelescopeFocuser with a current position and allowed range.

        Parameters
        ----------
        telescope_focuser : FocuserInterface
            The telescope focuser.
        telescope_pointer : TelescopePointer
            The telescope pointer.
        """
        self.focuser = focuser
        self.pointer = pointer

    def take_observation_at(self, focus_position: int, texp: float) -> ImageType:
        """
        Take an observation at a specific focus position with a given exposure time.

        Parameters
        ----------
        focus_position : int
            Desired focus position.
        texp : float
            Exposure time.

        Returns
        -------
        ImageType
            Resulting image.
        """        
        self.focuser.position = focus_position
        image = self.take_observation(texp=texp)

        return image

    def move_focuser_to_position(self, desired_position):
        self.focuser.position = desired_position

    @abstractmethod
    def take_observation(self, texp: float) -> ImageType:
        """
        Abstract method to take an observation with a specified exposure time.

        Parameters
        ----------
        texp : float
            Exposure time.

        Returns
        -------
        ImageType
            Resulting image.
        """
        pass

    def __repr__(self) -> str:
        return f"TelescopeInterface(self.focuser={self.focuser!r})"
