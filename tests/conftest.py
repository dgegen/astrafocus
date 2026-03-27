from pathlib import Path

import cabaret
import numpy as np
import pytest

from astrafocus.interface.simulation import DEFAULT_SOURCES

_RA = 10.68458
_DEC = 41.269


@pytest.fixture(scope="session")
def sky_image() -> np.ndarray:
    """Simulated night-sky image with 10 injected stars, returned as float64."""
    return (
        cabaret.Observatory()
        .generate_image(
            ra=_RA,
            dec=_DEC,
            sources=DEFAULT_SOURCES,
            exp_time=10,
            seed=42,
        )
        .astype(float)
    )


@pytest.fixture(scope="session")
def dark_image() -> np.ndarray:
    """Dark frame (no starlight) from the same observatory setup."""
    return (
        cabaret.Observatory()
        .generate_image(
            ra=_RA,
            dec=_DEC,
            sources=DEFAULT_SOURCES,
            exp_time=10,
            light=0,
            seed=42,
        )
        .astype(float)
    )


_FOCUS_POSITIONS = [9600, 9800, 10000, 10200, 10400]


@pytest.fixture(scope="session")
def fits_directory(tmp_path_factory) -> Path:
    """Directory of simulated FITS files spanning a focus sweep around best position."""
    directory = tmp_path_factory.mktemp("fits")
    for position in _FOCUS_POSITIONS:
        cabaret.Observatory(focuser={"position": position, "best_position": 10000}).generate_fits_image(
            ra=_RA,
            dec=_DEC,
            sources=DEFAULT_SOURCES,
            exp_time=10,
            seed=42,
            file_path=directory / f"focus_{position:05d}.fits",
            user_header={"FOCUSPOS": position},
        )
    return directory
