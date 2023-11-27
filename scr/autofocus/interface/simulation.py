import time
from typing import Optional, Tuple

import numpy as np

from autofocus.interface.focuser import FocuserInterface
from autofocus.interface.telescope import TelescopeInterface
from autofocus.interface.pointer import TrivialPointer
from autofocus.utils.load_fits_from_directory import load_fits_from_directory
from autofocus.utils.logger import configure_logger
from autofocus.utils.typing import ImageType

__all__ = ["get_telescope_simulation", "TelescopeSimulation"]

logger = configure_logger()

np.random.seed(42)


def get_telescope_simulation(
    path_to_data: str, sleep_flag: bool = False, seconds_per_step: float = 0.5
):
    """
    telescope_interface = get_telescope_simulation()
    """
    image_data, headers, focus_pos = load_fits_from_directory(path_to_data)
    telescope_simulation = TelescopeSimulation(
        current_position=focus_pos[3],
        allowed_range=(np.min(focus_pos), np.max(focus_pos)),
        sleep_flag=sleep_flag,
        seconds_per_step=seconds_per_step,
        image_data=image_data,
        headers=headers,
        focus_pos=focus_pos,
    )
    return telescope_simulation


class TelescopeFocuserSimulation(FocuserInterface):
    def __init__(
        self,
        current_position,
        allowed_range: Tuple[int, int],
        focus_pos: np.ndarray,
        seconds_per_step: float = 0.5,
        sleep_flag: bool = False,
    ):
        super().__init__(current_position=current_position, allowed_range=allowed_range)
        self.total_time_moving = 0.0

        # This attribute determines which position the simulator samples from
        self.position_to_sample = 0
        self.seconds_per_step = seconds_per_step
        self.sleep_flag = sleep_flag

        self.focus_pos = focus_pos

    def move_focuser_to_position(self, desired_position):
        """ """
        if desired_position not in self.focus_pos:
            left = np.where(self.focus_pos <= desired_position)[0][-1]
            p_bernoulli = (desired_position - self.focus_pos[left]) / (
                self.focus_pos[left + 1] - self.focus_pos[left]
            )
            move_to = np.random.binomial(n=1, p=p_bernoulli)

            self.position_to_sample = self.focus_pos[left + move_to]
        else:
            self.position_to_sample = desired_position

        # Moving the focuser by 10 microns takes about 5 second
        time_moving = self.seconds_per_step * abs(self.position - desired_position)
        if self.sleep_flag:
            time.sleep(time_moving)
        self.total_time_moving += time_moving


class TelescopeSimulation(TelescopeInterface):
    def __init__(
        self,
        current_position,
        allowed_range: Tuple[int, int],
        sleep_flag: bool = False,
        seconds_per_step: float = 0.5,
        image_data: Optional[list] = None,
        headers: Optional[list[ImageType]] = None,
        focus_pos: Optional[np.ndarray] = None,
        fits_path: Optional[str] = None,
    ):
        """
        Initialize the TelescopeSimulation with a current position and allowed range.

        Parameters
        ----------
        current_position : int
            The current focuser position in steps.
        allowed_range : tuple
            The range of allowed focuser steps (min_step, max_step).

        Examples
        --------
        telescope_simulation = TelescopeSimulation(
            current_position=focus_pos[3], allowed_range=tuple(focus_pos[[0, -1]])
        )
        image = telescope_simulation.take_observation_at(focus_position=focus_pos[12], texp=3.0)
        """
        # Load data
        if image_data is None or focus_pos is None:
            if fits_path is None:
                raise ValueError("Either provide image_data, headers and focus_pos or fits_path")
            image_data, headers, focus_pos = load_fits_from_directory(fits_path)

        self.image_data = image_data
        self.headers = headers
        self.focus_pos = focus_pos

        telescope_pointer = TrivialPointer()
        telescope_focuser = TelescopeFocuserSimulation(
            current_position=current_position,
            allowed_range=allowed_range,
            focus_pos=self.focus_pos,
            sleep_flag=sleep_flag,
            seconds_per_step=seconds_per_step,
        )
        super().__init__(focuser=telescope_focuser, pointer=telescope_pointer)
        self.total_time_exposing = 0.0
        self.sleep_flag = sleep_flag

        # start off with something realistic
        self.focuser.position_to_sample = self.focuser.position

    def take_observation(self, texp: float = 3.0):
        image = self.sample_an_observation(desired_position=self.focuser.position_to_sample)
        if self.sleep_flag:
            time.sleep(texp)
        self.total_time_exposing += texp
        return image

    @property
    def total_time(self):
        return self.total_time_exposing + self.focuser.total_time_moving

    @property
    def total_time_moving(self):
        return self.focuser.total_time_moving

    def sample_an_observation(self, desired_position: int, texp: float = 3.0):
        mask = np.where(self.focus_pos == desired_position)[0]

        return self.image_data[np.random.choice(mask)]
