import numpy as np
import pytest

from astrafocus.interface.simulation import ObservationalFocuserSimulation


@pytest.fixture
def focuser():
    # Not sorted, as the frames of a sweep in the other direction.
    return ObservationalFocuserSimulation(focus_pos=np.array([300, 100, 200, 200]), allowed_range=(0, 1000))


@pytest.mark.parametrize(
    "desired, expected",
    [(200, {200}), (150, {100, 200}), (290, {200, 300}), (50, {100}), (400, {300})],
)
def test_samples_nearest_observed_positions(focuser, desired, expected):
    sampled = set()
    for _ in range(50):
        focuser.move_focuser_to_position(desired)
        sampled.add(int(focuser.position_to_sample))
    assert sampled == expected
