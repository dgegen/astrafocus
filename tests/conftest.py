import cabaret
import numpy as np
import pytest
from cabaret.sources import Sources

_RA = 10.68458
_DEC = 41.269

_SOURCES = Sources.from_arrays(
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


@pytest.fixture(scope="session")
def sky_image() -> np.ndarray:
    """Simulated night-sky image with 10 injected stars, returned as float64."""
    return (
        cabaret.Observatory()
        .generate_image(
            ra=_RA,
            dec=_DEC,
            sources=_SOURCES,
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
            sources=_SOURCES,
            exp_time=10,
            light=0,
            seed=42,
        )
        .astype(float)
    )
