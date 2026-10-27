"""
Regression tests on the SPECULOOS Callisto autofocus runs of 2026-10-23 (night of 2026-10-22).

The frames are not in the repository. Set `path_to_callisto_20261023` in `tests/config.yaml` to the
directory with the 63 FITS frames to run these tests. Each run is a reference exposure at the start
position, followed by a sweep of 20 frames over start +- 1000, downwards. The focus positions used
are the ones astrafocus commanded; `FOCUSPOS` in the headers differs from them by 25 to 60 steps.

- Run 3 has a bright star. Before the sky was subtracted, the HFR was about 45.4 at every position
  and the autofocuser accepted a random best focus. The independent reference is about 7010 in
  `FOCUSPOS` units, which is about 6960 in commanded units.
  The same run, cropped to 320 x 320 px, checks that a donut that fills most of a row does not
  raise the sky.
- Run 1 has only faint stars that vanish far from focus. The run must fail and move the focuser
  back to its start position.
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from utils import ConfigTests

from astrafocus.autofocuser import AnalyticResponseAutofocuser
from astrafocus.interface.simulation import ObservationBasedDeviceSimulator
from astrafocus.star_size_focus_measure_operators import GaussianStarFocusMeasure, HFRStarFocusMeasure

DATA_DIR = ConfigTests().get().get("path_to_callisto_20261023")
pytestmark = pytest.mark.skipif(
    DATA_DIR is None or not Path(DATA_DIR).is_dir(),
    reason="No path_to_callisto_20261023 in tests/config.yaml.",
)

REFERENCE_FOCUS = 6960
STEP = 105  # step of the commanded sweep
HALF_RANGE = 1000
RUN_GAP = 120  # seconds between frames that separate runs
BRIGHT_STAR = (500, 632)  # (x, y) in run 3
TIME_PATTERN = re.compile(r"_(\d{8}_\d{6}\.\d+)\.fits$")


def load_run(run_number):
    """Return the images (reference first), the commanded focus positions and the exposure time."""
    frames = sorted(
        (datetime.strptime(match.group(1), "%Y%m%d_%H%M%S.%f"), path)
        for path in Path(DATA_DIR).glob("*.fits")
        if (match := TIME_PATTERN.search(path.name))
    )
    runs = [[frames[0][1]]]
    for (previous, _), (current, path) in zip(frames, frames[1:]):
        if (current - previous).total_seconds() > RUN_GAP:
            runs.append([])
        runs[-1].append(path)

    images, headers = [], []
    for path in runs[run_number - 1]:
        with fits.open(path) as hdul:
            images.append(hdul[0].data.astype(np.float32))
            headers.append(hdul[0].header)
    start = int(round(headers[0]["FOCUSPOS"]))
    sweep = np.round(np.linspace(start - HALF_RANGE, start + HALF_RANGE, len(images) - 1)).astype(int)
    positions = np.array([start, *sweep[::-1]])
    return images, positions, float(headers[0]["EXPTIME"])


@pytest.fixture(scope="module")
def run1():
    return load_run(1)


@pytest.fixture(scope="module")
def run3():
    return load_run(3)


def crop(images, half_size, centre=BRIGHT_STAR):
    x, y = centre
    return [image[y - half_size : y + half_size, x - half_size : x + half_size] for image in images]


def replay(images, positions, exposure_time, focus_measure_operator):
    """Run the autofocus on the recorded frames. The first frame is the reference exposure."""
    order = np.argsort(positions)
    simulator = ObservationBasedDeviceSimulator(
        image_data=[images[i] for i in order],
        headers=[None] * len(images),
        focus_pos=positions[order],
        current_position=int(positions[0]),
    )
    autofocuser = AnalyticResponseAutofocuser(
        autofocus_device_manager=simulator,
        exposure_time=exposure_time,
        focus_measure_operator=focus_measure_operator,
        n_steps=(len(positions) - 1,),
        n_exposures=(1,),
        search_range=(int(positions[1:].min()), int(positions[1:].max())),
        decrease_search_range=False,
    )
    np.testing.assert_array_equal(
        autofocuser.integer_linspace(*autofocuser.search_range, len(positions) - 1), np.sort(positions[1:])
    )
    success = autofocuser.run()
    return success, autofocuser, simulator


def assert_finds_best_focus(success, autofocuser, simulator):
    assert success
    assert abs(autofocuser.best_focus_position - REFERENCE_FOCUS) <= STEP
    assert simulator.focuser.position == autofocuser.best_focus_position
    _, focus_measure = autofocuser.get_focus_record()
    assert focus_measure.max() > 5 * focus_measure.min()


@pytest.mark.parametrize("focus_measure_operator", [HFRStarFocusMeasure, GaussianStarFocusMeasure])
def test_bright_star_run_finds_best_focus(run3, focus_measure_operator):
    assert_finds_best_focus(*replay(*run3, focus_measure_operator))


@pytest.mark.parametrize("focus_measure_operator", [HFRStarFocusMeasure, GaussianStarFocusMeasure])
def test_bright_star_run_finds_best_focus_in_small_frames(run3, focus_measure_operator):
    # Far from focus, the donut is about 200 px wide and fills most rows of this 320 px crop. Without
    # masking the star, the sky of each row includes its light (see `prepare_image`).
    images, positions, exposure_time = run3
    assert_finds_best_focus(*replay(crop(images, 160), positions, exposure_time, focus_measure_operator))


@pytest.mark.parametrize("focus_measure_operator", [HFRStarFocusMeasure, GaussianStarFocusMeasure])
def test_faint_star_run_fails_and_resets_focuser(run1, focus_measure_operator):
    success, autofocuser, simulator = replay(*run1, focus_measure_operator)

    assert not success
    assert autofocuser.best_focus_position is None
    assert simulator.focuser.position == autofocuser.initial_position
