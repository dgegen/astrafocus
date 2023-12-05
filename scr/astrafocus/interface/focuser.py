from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class FocuserInterface(ABC):
    """
    A class to manage the focus of a telescope.

    Attributes
    ----------
    position : int
        The current focuser position in steps.
    allowed_range : tuple
        The range of allowed focuser steps (min_step, max_step).

    Methods
    -------
    move_by_steps(steps_to_move)
        Move the focuser relative to the current position by n steps.
    """

    def __init__(self, current_position, allowed_range: Tuple[int, int]):
        """
        Initialize the TelescopeFocuser with a current position and allowed range.

        Parameters
        ----------
        current_position : int
            The current focuser position in steps.
        allowed_range : tuple
            The range of allowed focuser steps (min_step, max_step).
        """
        self._current_position = current_position
        self.allowed_range = allowed_range
        self.validate_allowed_range()

        if not self.is_within_range(self._current_position):
            raise ValueError(
                f"Initial current position {self._current_position} "
                "is outside the provided allowed range {self.allowed_range}."
            )

    @abstractmethod
    def move_focuser_to_position(self, new_position):
        pass

    @property
    def position(self):
        return self._current_position

    @position.setter
    def position(self, new_position):
        """
        Set the current focuser position.

        Parameters
        ----------
        position : int
            The desired focuser position in steps.

        Raises
        ------
        ValueError
            If the desired position is outside the allowed range.
        """
        self.validate_desired_position(new_position)
        self.move_focuser_to_position(new_position)
        self._current_position = new_position


    def move_by_steps(self, steps_to_move):
        """
        Move the focuser relative to the current position by n steps.

        Parameters
        ----------
        steps_to_move : int
            The number of steps to move relative to the current position.

        Raises
        ------
        ValueError
            If moving relative exceeds the allowed range.
        """
        desired_position = self._current_position + steps_to_move
        self.validate_desired_position(desired_position)
        self._current_position = desired_position

    def validate_desired_position(self, desired_position):
        if not isinstance(desired_position, (int, np.integer)):
            raise ValueError("All focuser positions must be integers.")

        if not self.is_within_range(desired_position):
            min_step, max_step = self.allowed_range
            raise ValueError(
                f"Moving to the desired position, {desired_position}, "
                f"would exceed the allowed range, ({min_step}, {max_step})."
            )

    def is_within_range(self, desired_position):
        """
        Check whether a desired focuser position is within the allowed range.

        Parameters
        ----------
        desired_position : int
            The desired focuser position to check.

        Returns
        -------
        bool
            True if the desired position is within the allowed range, False otherwise.
        """
        min_step, max_step = self.allowed_range
        return min_step <= desired_position <= max_step

    def validate_allowed_range(self):
        """
        Validate the allowed range provided during initialization.

        Raises
        ------
        ValueError
            If the allowed range is not a tuple, list, or numpy array, or if any items
            in the range are not integers, or if the range does not consist of two integers.
        """
        if not isinstance(self.allowed_range, (tuple, list, np.ndarray)):
            raise ValueError("The allowed range must be a tuple, list, or numpy array.")

        self.allowed_range = tuple(self.allowed_range)

        if any(not isinstance(item, (int, np.integer)) for item in self.allowed_range):
            raise ValueError("The boundaries of the allowed range must be integers.")

        if not len(self.allowed_range) == 2:
            raise ValueError(
                "The allowed range should consist of two integers (min_step, max_step)."
            )

        if not (self.allowed_range[0] < self.allowed_range[1]):
            raise ValueError(
                "max_step must be greater than min_step in the allowed range (min_step, max_step)."
            )

    def __repr__(self) -> str:
        return (
            "FocuserInterface("
            f"current_position={self.position!r}, allowed_range={self.allowed_range!r}"
            ")"
        )


class TrivialFocuser(FocuserInterface):
    """
    Trivial implementation to set the telescope focuser for testing purposes.
    """
    def __init__(self, current_position, allowed_range=Tuple[int, int]):
        super().__init__(current_position=current_position, allowed_range=allowed_range)

    def move_focuser_to_position(self, new_position):
        pass
