"""Synthetic CMOS-like frames for testing star-size focus measures.

The frames have a high sky level, row banding, hot pixels, and stars whose profile goes from a
sharp Gaussian at best focus to a large donut far from focus.
"""

import numpy as np
from scipy import ndimage

SKY_LEVEL = 3000.0
READ_NOISE = 10.0
HOT_PIXEL_FRACTION = 0.003
BEST_FOCUS = 10_000
SEEING_SIGMA = 1.5  # pixels
DONUT_RADIUS_PER_STEP = 55 / 1000  # outer donut radius (pixels) per focuser step from best focus
DONUT_INNER_FRACTION = 0.35  # central obstruction, as a fraction of the outer radius
STAR_POSITIONS = ((130.0, 140.0), (380.0, 120.0), (125.0, 385.0), (390.0, 370.0))
STAR_FLUX = 3e6
SHAPE = (512, 512)


def star_profile(shape, x, y, flux, donut_radius, sigma=SEEING_SIGMA, oversample=4):
    """Render a star as an annulus of outer radius `donut_radius`, blurred by a Gaussian."""
    if donut_radius < 1:
        yy, xx = np.indices(shape, dtype=float)
        profile = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        return flux * profile / profile.sum()

    fine_shape = (shape[0] * oversample, shape[1] * oversample)
    yy, xx = (np.indices(fine_shape, dtype=float) + 0.5) / oversample - 0.5
    r = np.hypot(xx - x, yy - y)
    annulus = ((r <= donut_radius) & (r >= DONUT_INNER_FRACTION * donut_radius)).astype(float)
    annulus = annulus.reshape(shape[0], oversample, shape[1], oversample).mean(axis=(1, 3))
    profile = ndimage.gaussian_filter(annulus, sigma)
    return flux * profile / profile.sum()


def donut_radius_at(focus_position, best_focus=BEST_FOCUS):
    return DONUT_RADIUS_PER_STEP * abs(focus_position - best_focus)


def make_frame(
    focus_position=BEST_FOCUS,
    best_focus=BEST_FOCUS,
    shape=SHAPE,
    star_positions=STAR_POSITIONS,
    star_flux=STAR_FLUX,
    sigma=SEEING_SIGMA,
    shift=(0.0, 0.0),
    sky_level=SKY_LEVEL,
    hot_pixel_fraction=HOT_PIXEL_FRACTION,
    banding_amplitude=60.0,
    seed=0,
):
    """
    Make one frame. The noise, banding and hot pixels depend only on `seed`, so hot pixels stay at
    the same place in every frame of a sweep, as on a real sensor.
    """
    rng = np.random.default_rng(seed)
    donut_radius = donut_radius_at(focus_position, best_focus)

    image = np.zeros(shape)
    for x, y in star_positions:
        image += star_profile(shape, x + shift[0], y + shift[1], star_flux, donut_radius, sigma)

    image += sky_level
    image = rng.poisson(image).astype(float)
    image += rng.normal(0, READ_NOISE, size=shape)

    # Row banding: a slow wave plus a random offset per row.
    rows = np.arange(shape[0])
    banding = banding_amplitude * np.sin(2 * np.pi * rows / 37.0)
    banding += rng.normal(0, banding_amplitude / 2, size=shape[0])
    image += banding[:, None]

    hot_x, hot_y, hot_values = hot_pixels(shape, hot_pixel_fraction)
    image[hot_y, hot_x] += hot_values

    return image


def hot_pixels(shape=SHAPE, hot_pixel_fraction=HOT_PIXEL_FRACTION):
    """Positions and values of the hot pixels. They are the same in every frame, as on a real sensor."""
    hot_rng = np.random.default_rng(12345)
    n_hot = int(hot_pixel_fraction * shape[0] * shape[1])
    hot_y = hot_rng.integers(0, shape[0], n_hot)
    hot_x = hot_rng.integers(0, shape[1], n_hot)
    return hot_x, hot_y, hot_rng.uniform(2_000, 20_000, n_hot)
