import time
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from astrafocus.interface.camera import CameraInterface
from astrafocus.interface.focuser import FocuserInterface
from astrafocus.interface.telescope import TrivialTelescope
from astrafocus.interface.device_manager import AutofocusDeviceManager
from astrafocus.utils.load_fits_from_directory import load_fits_from_directory
from astrafocus.utils.logger import configure_logger
from astrafocus.utils.typing import ImageType

__all__ = ["ObservationBasedDeviceSimulator"]

logger = configure_logger()

np.random.seed(42)


class FocuserSimulation(FocuserInterface):
    def __init__(
        self,
        current_position,
        allowed_range: Tuple[int, int],
        seconds_per_step: float = 0.5,
        sleep_flag: bool = False,
    ):
        super().__init__(current_position=current_position, allowed_range=allowed_range)
        self.total_time_moving = 0.0

        self.seconds_per_step = seconds_per_step
        self.sleep_flag = sleep_flag

    def move_focuser_to_position(self, desired_position):
        time_moving = self.seconds_per_step * abs(self.position - desired_position)
        if self.sleep_flag:
            time.sleep(time_moving)
        self.total_time_moving += time_moving

    @property
    def position_to_sample(self):
        """
        The focus position to use when simulating an exposure.

        This attribute is used to account for the fact that some simulators, like the
        ObservationalFocuserSimulation, sample from set of focus positions, which can be a proper
        subset of the integers within the allowed range. In these cases, the position to sample
        from is not necessarily the same as the current position, but might be the closest
        focus position to the current position that was observed.

        It can further be used to simulate a focuser that is not perfectly accurate, by setting
        the position to sample from to a different value than the current position.

        If this attribute is not set, it defaults to the current position.
        """
        return getattr(self, "_position_to_sample", self.position)

    @position_to_sample.setter
    def position_to_sample(self, value):
        if hasattr(self, "_position_to_sample"):
            self._position_to_sample = value
        else:
            self.position = value


class ObservationalFocuserSimulation(FocuserSimulation):
    """Simulates a focuser based on a set of observations."""

    def __init__(
        self,
        focus_pos: np.ndarray,
        current_position: Optional[int] = None,
        allowed_range: Optional[Tuple[int, int]] = None,
        seconds_per_step: float = 0.5,
        sleep_flag: bool = False,
    ):
        if allowed_range is None:
            allowed_range = (np.min(focus_pos), np.max(focus_pos))
        if current_position is None:
            current_position = focus_pos[0]

        super().__init__(
            current_position=current_position,
            allowed_range=allowed_range,
            seconds_per_step=seconds_per_step,
            sleep_flag=sleep_flag,
        )
        # Array of focus positions at which exposures were taken
        self.focus_pos = focus_pos

        # This attribute determines which position the simulator samples from
        self._position_to_sample = self.position

    def move_focuser_to_position(self, desired_position):
        if desired_position not in self.focus_pos:
            left = np.where(self.focus_pos <= desired_position)[0][-1]
            p_bernoulli = (desired_position - self.focus_pos[left]) / (
                self.focus_pos[left + 1] - self.focus_pos[left]
            )
            move_to = np.random.binomial(n=1, p=p_bernoulli)

            self.position_to_sample = self.focus_pos[left + move_to]
        else:
            self.position_to_sample = desired_position

        super().move_focuser_to_position(self.position_to_sample)


class CameraSimulation(CameraInterface):
    def __init__(
        self,
        focuser: FocuserSimulation,
        sleep_flag: bool = False,
    ):
        super().__init__()
        self.focuser = focuser
        self.total_time_exposing = 0.0
        self.sleep_flag = sleep_flag

    def perform_exposure(self, texp: float = 3.0):
        """Capture an observation at the specified focal position."""
        image = self.sample_an_observation(
            desired_position=self.focuser.position_to_sample, texp=texp
        )
        if self.sleep_flag:
            time.sleep(texp)
        self.total_time_exposing += texp
        return image

    @abstractmethod
    def sample_an_observation(self, desired_position: int, texp: float = 3.0):
        pass


class ObservationalCameraSimulation(CameraSimulation):
    """Simulates a camera based on a set of observations.

    This simulation takes observations from a set of image data and associated
    headers. It can optionally introduce a sleep delay after each observation.
    The `capture_observation` method captures an image, and the
    `get_observation_at_position` method selects an observation based on the desired
    focal position.
    """

    def __init__(
        self,
        image_data: np.ndarray,
        focuser: np.ndarray,
        sleep_flag: bool = False,
    ):
        super().__init__(focuser=focuser, sleep_flag=sleep_flag)
        self.image_data = image_data

    def sample_an_observation(self, desired_position: int, texp: float = 3.0):
        """Get an observation at the specified focal position."""
        mask = np.where(self.focuser.focus_pos == desired_position)[0]

        return self.image_data[np.random.choice(mask)]

    def perform_exposure(self, texp: float = 3.0):
        """Capture an observation at the specified focal position."""
        image = self.sample_an_observation(
            desired_position=self.focuser.position_to_sample, texp=texp
        )
        if self.sleep_flag:
            time.sleep(texp)
        self.total_time_exposing += texp
        return image


class AutofocusDeviceSimulator(AutofocusDeviceManager):
    def __init__(
        self,
        camera: CameraSimulation,
        focuser: FocuserSimulation,
        telescope=TrivialTelescope(),
        sleep_flag: bool = False,
    ):
        """Initialize the AutofocusDeviceSimulator with a camera, focuser and telescope."""
        super().__init__(camera=camera, focuser=focuser, telescope=telescope)
        self._sleep_flag = sleep_flag

    @property
    def sleep_flag(self):
        return self._sleep_flag

    @sleep_flag.setter
    def sleep_flag(self, value):
        self._sleep_flag = value
        self.camera.sleep_flag = value
        self.focuser.sleep_flag = value

    @property
    def total_time(self):
        return self.camera.total_time_exposing + self.focuser.total_time_moving

    @property
    def total_time_moving(self):
        return self.focuser.total_time_moving

    @property
    def total_time_exposing(self):
        return self.camera.total_time_exposing


class ObservationBasedDeviceSimulator(AutofocusDeviceSimulator):
    def __init__(
        self,
        current_position: Optional[int] = None,
        allowed_range: Optional[Tuple[int, int]] = None,
        sleep_flag: bool = False,
        seconds_per_step: float = 0.5,
        image_data: Optional[list] = None,
        headers: Optional[list[ImageType]] = None,
        focus_pos: Optional[np.ndarray] = None,
        fits_path: Optional[str] = None,
    ):
        """
        Initialize the AutofocusDeviceSimulator with a current position and allowed range.

        Parameters
        ----------
        current_position : int
            The current focuser position in steps.
        allowed_range : tuple
            The range of allowed focuser steps (min_step, max_step).

        Examples
        --------
        telescope_simulation = AutofocusDeviceSimulator(
            current_position=focus_pos[3], allowed_range=tuple(focus_pos[[0, -1]])
        )
        image = telescope_simulation.perform_exposure_at(focus_position=focus_pos[12], texp=3.0)
        """
        # Load data
        if image_data is None or focus_pos is None:
            if fits_path is None:
                raise ValueError("Either provide image_data, headers and focus_pos or fits_path")
            image_data, headers, focus_pos = load_fits_from_directory(fits_path)

        self.image_data = image_data
        self.headers = headers
        self.focus_pos = focus_pos

        focuser = ObservationalFocuserSimulation(
            focus_pos=self.focus_pos,
            current_position=current_position,
            allowed_range=allowed_range,
            sleep_flag=sleep_flag,
            seconds_per_step=seconds_per_step,
        )

        camera = ObservationalCameraSimulation(
            image_data=image_data, focuser=focuser, sleep_flag=sleep_flag
        )
        super().__init__(camera=camera, focuser=focuser, telescope=None, sleep_flag=sleep_flag)

        # start off with something realistic
        self.focuser.position_to_sample = self.focuser.position
