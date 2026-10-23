import numpy as np


def focus_curve_snr(focus_pos: np.ndarray, focus_measure: np.ndarray) -> float:
    """
    Ratio of the spread of the focus measure to its noise.

    The spread is the range of the measures, sorted by focus position, after a 3-point running
    median. The noise comes from the median absolute deviation of the second differences, which
    a smooth curve does not change. Returns infinity with fewer than 5 valid points.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = np.linspace(-1000, 1000, 20)
    >>> v_curve = 2 + np.abs(x) / 25 + rng.normal(0, 0.1, x.size)
    >>> bool(focus_curve_snr(x, v_curve) > 8)
    True
    >>> flat = 45.8 + rng.normal(0, 0.003, x.size)
    >>> bool(focus_curve_snr(x, flat) < 8)
    True
    """
    focus_pos = np.asarray(focus_pos, dtype=float)
    focus_measure = np.asarray(focus_measure, dtype=float)
    valid = np.isfinite(focus_pos) & np.isfinite(focus_measure)
    y = focus_measure[valid][np.argsort(focus_pos[valid], kind="stable")]
    if y.size < 5:
        return np.inf

    running_median = np.median(np.lib.stride_tricks.sliding_window_view(y, 3), axis=1)
    spread = np.ptp(np.concatenate([y[:1], running_median, y[-1:]]))

    second_diff = np.diff(y, 2)
    noise = 1.4826 * np.median(np.abs(second_diff - np.median(second_diff))) / np.sqrt(6)
    if noise == 0:
        return np.inf if spread > 0 else 0.0
    return float(spread / noise)
