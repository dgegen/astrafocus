import numpy as np
import pytest

from astrafocus import FocusMeasureOperatorRegistry
from astrafocus.interface.simulation import CabaretDeviceSimulator

IMAGE_OPERATORS = [name for name in FocusMeasureOperatorRegistry.list() if name not in ("hfr", "gauss")]


@pytest.fixture(scope="module")
def uint16_image():
    return CabaretDeviceSimulator.generate_image(seed=0)


@pytest.mark.parametrize("name", IMAGE_OPERATORS)
@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32])
def test_value_does_not_depend_on_dtype(name, dtype, uint16_image):
    # Differences of unsigned integers wrap around, and OpenCV doesn't support every input type.
    assert uint16_image.dtype == np.uint16
    operator = FocusMeasureOperatorRegistry.from_name(name)()
    expected = operator(uint16_image.astype(np.float64))
    assert operator(uint16_image.astype(dtype)) == pytest.approx(expected, rel=1e-9)
