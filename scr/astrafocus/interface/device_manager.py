from abc import ABC, abstractmethod
from typing import Optional

from astrafocus.interface.camera import CameraInterface, TrivialCamera
from astrafocus.interface.focuser import FocuserInterface, TrivialFocuser
from astrafocus.interface.telescope import TelescopeInterface, TrivialTelescope
from astrafocus.utils.typing import ImageType


class AutofocusDeviceManager(ABC):
    """
    Abstract base class representing a telescope interface.

    Parameters
    ----------
    camera : CameraInterface
        The interface with the camera.
    focuser : FocuserInterface
        The interface with the focuser.
    telescope : TelescopeInterface
        The interface with the telescope, responsible for slewing to a position on the sky.

    Methods
    -------
    perform_exposure_at(focus_position: int, texp: float) -> ImageType
        Take an observation at a specific focus position with a given exposure time.
    move_focuser_to_position(desired_position)
        Move the focuser to a desired position.
    check_conditions() -> bool
        Check if observation conditions are good enough to take exposures. Default is True.
        The implementation of this method is optional, although it is recommended to provide one.
        This function will be used to interrupt the autofocus process if the conditions get bad.

    Examples
    --------
    # Create a trivial autofocus device manager.
    >>> from astrafocus.interface.device_manager import AutofocusDeviceManager
    >>> from astrafocus.interface.camera import TrivialCamera
    >>> from astrafocus.interface.focuser import TrivialFocuser
    >>> from astrafocus.interface.telescope import TrivialTelescope
    >>> autofocus_device_manager = AutofocusDeviceManager(
        camera=TrivialCamera(),
        focuser=TrivialFocuser(current_position=0, allowed_range=(0, 1000)),
        telescope=TrivialTelescope()
    )
    """

    def __init__(
        self,
        camera: CameraInterface,
        focuser: FocuserInterface,
        telescope: Optional[TelescopeInterface] = TrivialTelescope(),
    ):
        """
        Initialize the TelescopeFocuser with a current position and allowed range.

        Parameters
        ----------
        camera : CameraInterface
            The interface with the camera.
        focuser : FocuserInterface
            The interface with the focuser.
        telescope : TelescopeInterface
            The interface with the telescope, responsible for slewing to a position on the sky.
        """
        self.camera = camera
        self.focuser = focuser
        self.telescope = telescope

    def perform_exposure_at(self, focus_position: int, texp: float) -> ImageType:
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
        image = self.camera.perform_exposure(texp=texp)

        return image

    def move_focuser_to_position(self, desired_position):
        self.focuser.position = desired_position
    
    def check_conditions(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            f"AutofocusDeviceManager(self.camera={self.camera!r}, "
            f"self.focuser={self.focuser!r}, self.telescope={self.telescope!r})"
        )


class TrivialAutofocusDeviceManager(AutofocusDeviceManager):
    """
    A trivial telescope interface for testing purposes.
    """

    def __init__(
        self,
        camera: CameraInterface = TrivialCamera(),
        focuser: FocuserInterface = TrivialFocuser(current_position=0, allowed_range=(0, 1000)),
        telescope: TelescopeInterface = TrivialTelescope(),
    ):
        super().__init__(camera=camera, focuser=focuser, telescope=telescope)

    def __repr__(self) -> str:
        return (
            f"TrivialAutofocusDeviceManager(self.camera={self.camera!r}, "
            f"self.focuser={self.focuser!r}, self.telescope={self.telescope!r})"
        )
