import os
import time
from abc import abstractmethod
from datetime import datetime

import cabaret
import numpy as np
from astropy.io import fits
from cabaret.sources import Sources

from astrafocus.interface.camera import CameraInterface
from astrafocus.interface.device_manager import AutofocusDeviceManager
from astrafocus.interface.focuser import FocuserInterface
from astrafocus.interface.telescope import TrivialTelescope
from astrafocus.utils.fits import load_fits_with_focus_pos_from_directory
from astrafocus.utils.typing import ImageType

__all__ = ["ObservationBasedDeviceSimulator"]

DEFAULT_SOURCES = Sources.from_arrays(
    ra=np.array(
        [
            10.63950247,
            10.6880729,
            10.70349581,
            10.71333998,
            10.70022208,
            10.68780207,
            10.63804045,
            10.73974676,
            10.71181048,
            10.67584817,
        ]
    ),
    dec=np.array(
        [
            41.26393165,
            41.22524785,
            41.25357386,
            41.29943347,
            41.26019689,
            41.31717482,
            41.27468757,
            41.2942209,
            41.29130279,
            41.26275058,
        ]
    ),
    fluxes=np.array(
        [
            248801.90860538,
            63459.13500958,
            37018.39316537,
            19189.9952933,
            16366.89603078,
            13204.99311425,
            10040.65065901,
            8150.59415653,
            6937.47387072,
            6500.26581163,
        ]
    ),
)


class FocuserSimulation(FocuserInterface):
    def __init__(
        self,
        current_position,
        allowed_range: tuple[int, int],
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
        current_position: int | None = None,
        allowed_range: tuple[int, int] | None = None,
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
        image = self.sample_an_observation(desired_position=self.focuser.position_to_sample, texp=texp)
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
        focuser: ObservationalFocuserSimulation,
        sleep_flag: bool = False,
        save_path: str | None = None,
    ):
        super().__init__(focuser=focuser, sleep_flag=sleep_flag)
        self.image_data = image_data
        self.save_path = save_path

    def sample_an_observation(self, desired_position: int, texp: float = 3.0):
        """Get an observation at the specified focal position."""
        mask = np.where(self.focuser.focus_pos == desired_position)[0]

        return self.image_data[np.random.choice(mask)]

    def perform_exposure(self, texp: float = 3.0):
        """Capture an observation at the specified focal position."""
        image = self.sample_an_observation(desired_position=self.focuser.position_to_sample, texp=texp)
        if self.sleep_flag:
            time.sleep(texp)
        self.total_time_exposing += texp

        if self.save_path is not None:
            file_name = f"{datetime.now().strftime('%Y-%m-%dT%H%M%S%f')}.fits"
            fits.writeto(os.path.join(self.save_path, file_name), image, overwrite=True)

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
        current_position: int | None = None,
        allowed_range: tuple[int, int] | None = None,
        sleep_flag: bool = False,
        seconds_per_step: float = 0.5,
        image_data: list | None = None,
        headers: list[ImageType] | None = None,
        focus_pos: np.ndarray | None = None,
        fits_path: str | None = None,
        save_path: str | None = None,
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
            image_data, headers, focus_pos = load_fits_with_focus_pos_from_directory(fits_path)
            assert focus_pos.size > 0, f"No focus positions found in fits files {fits_path}"

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
            image_data=image_data,
            focuser=focuser,
            sleep_flag=sleep_flag,
            save_path=save_path,
        )
        super().__init__(camera=camera, focuser=focuser, telescope=None, sleep_flag=sleep_flag)

        # start off with something realistic
        self.focuser.position_to_sample = self.focuser.position


class CabaretCameraSimulator(CameraSimulation):
    def __init__(
        self,
        focuser: FocuserSimulation,
        observatory: cabaret.Observatory | None = None,
        sources: cabaret.Sources | None = None,
        sleep_flag: bool = False,
        save_path: str | None = None,
    ):
        super().__init__(focuser=focuser, sleep_flag=sleep_flag)
        cabaret_focuser = cabaret.Focuser(
            position=11_000,
            best_position=10_000,
            scale=100,
        )
        self.observatory = (
            observatory if observatory is not None else cabaret.Observatory(focuser=cabaret_focuser)
        )
        self.save_path = save_path
        self.sources = sources if sources is not None else cabaret.Sources.get_test_sources()

    def sample_an_observation(self, desired_position: int, texp: float = 3.0):
        """Get an observation at the specified focal position."""
        ra, dec = self.sources.center
        self.observatory.focuser.position = desired_position
        image = self.observatory.generate_image(
            ra=ra,
            dec=dec,
            exp_time=texp,
            sources=self.sources,
        )
        return image


class CabaretDeviceSimulator(AutofocusDeviceSimulator):
    def __init__(
        self,
        current_position: int = 9_550,
        allowed_range: tuple[int, int] = (9_500, 10_500),
        sleep_flag: bool = False,
        seconds_per_step: float = 0.5,
        observatory: cabaret.Observatory | None = None,
        sources: cabaret.Sources | None = None,
        save_path: str | None = None,
    ):
        focuser = FocuserSimulation(
            current_position=current_position,
            allowed_range=allowed_range,
            sleep_flag=sleep_flag,
            seconds_per_step=seconds_per_step,
        )
        camera = CabaretCameraSimulator(
            focuser=focuser,
            observatory=observatory,
            sources=sources,
            sleep_flag=sleep_flag,
            save_path=save_path,
        )
        super().__init__(camera=camera, focuser=focuser, telescope=None, sleep_flag=sleep_flag)

    @classmethod
    def default(cls, **kwargs) -> "CabaretDeviceSimulator":
        """Return a simulator pre-loaded with a fixed set of sources near M31.

        The sources are a catalogue of 10 stars with known fluxes, so no
        Gaia query is performed and results are fully reproducible.

        Parameters
        ----------
        **kwargs
            Forwarded to :class:`CabaretDeviceSimulator`.
        """
        return cls(sources=DEFAULT_SOURCES, **kwargs)

    @classmethod
    def generate_image(cls, focus_position: int = 10_000, texp: float = 1.0) -> np.ndarray:
        """Generate a single synthetic sky image at the given focus position.

        Uses :meth:`default` sources so no Gaia query is required.

        Parameters
        ----------
        focus_position : int
            Focuser position at which to render the image.
        texp : float
            Exposure time in seconds.

        Returns
        -------
        numpy.ndarray
            Simulated 2-D image array.

        Examples
        --------
        >>> from astrafocus.interface.simulation import CabaretDeviceSimulator
        >>> image = CabaretDeviceSimulator.generate_image()
        >>> image.ndim
        2
        """
        return cls.default().camera.sample_an_observation(desired_position=focus_position, texp=texp)
