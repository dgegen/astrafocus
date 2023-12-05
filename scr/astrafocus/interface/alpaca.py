import time
import astropy
import alpaca
from alpaca.focuser import *
from alpaca.camera import *
from alpaca.telescope import *

from astrafocus.interface.camera import CameraInterface
from astrafocus.interface.focuser import FocuserInterface
from astrafocus.interface.telescope import TelescopeInterface
from astrafocus.interface.device_manager import AutofocusDeviceManager

__all__ = ['AlpacaAutofocusDeviceManager']

class AlpacaCamera(CameraInterface):
    def __init__(self, alpaca_camera: alpaca.camera.Camera):
        self.alpaca_camera = alpaca_camera
        super().__init__()

    def perform_exposure(self, texp: float):
        """
        Calling this function should take as long as it takes to make the observation.
        """
        self.alpaca_camera.StartExposure(Duration=texp, Light=False)

        # Wait while the camera is recording
        while not (self.alpaca_camera.ImageReady):
            time.sleep(min(0.1, texp / 10))

        image = self.alpaca_camera.ImageArray
        return image


class AlpacaTelescope(TelescopeInterface):
    def __init__(self, alpaca_telescope: alpaca.telescope.Telescope):
        self.alpaca_telescope = alpaca_telescope
        super().__init__()

    def set_telescope_position(self, coordinates: astropy.coordinates.SkyCoord, hard_timeout: float = 120):
        """
        Calling this function should take as long as it takes to move the telescope to the desired position.
        """
        self.alpaca_telescope.SlewToCoordinatesAsync(
            RightAscension=coordinates.ra.hour, Declination=coordinates.dec.deg
        )

        # Wait for slew to finish
        start_time = time.time()
        while self.alpaca_telescope.Slewing is True:
            if time.time() - start_time > hard_timeout:
                raise TimeoutError('Slew timeout')

            time.sleep(1)


class AlpacaFocuser(FocuserInterface):
    def __init__(self, alpaca_focuser: alpaca.focuser.Focuser):
        if not alpaca_focuser.Absolute:
            raise ValueError("Focuser must be absolute for autofocusing to work.")

        self.alpaca_focuser = alpaca_focuser

        current_position = self.get_current_position()
        allowed_range = (0, alpaca_focuser.MaxStep)
        super().__init__(current_position=current_position, allowed_range=allowed_range)

    def move_focuser_to_position(self, new_position: int):
        """
        Calling this function should take as long as it takes to move the focuser to the desired position.
        """
        self.alpaca_focuser.Move(new_position)
        while self.alpaca_focuser.IsMoving:
            time.sleep(0.01)

        return None

    def get_current_position(self):
        return self.alpaca_focuser.Position


class AlpacaAutofocusDeviceManager(AutofocusDeviceManager):
    def __init__(
        self,
        alpaca_camera: alpaca.camera.Camera,
        alpaca_focuser: alpaca.focuser.Focuser,
        alpaca_telescope: alpaca.telescope.Telescope,
    ):
        camera = AlpacaCamera(alpaca_camera)
        focuser = AlpacaFocuser(alpaca_focuser)
        telescope = AlpacaTelescope(alpaca_telescope)
        super().__init__(camera=camera, focuser=focuser, telescope=telescope)
