# Operator Reference

This page lists the built-in focus-measure operators, recommended use (coarse vs fine), important parameters, and common failure modes.

| Key | Operator | Recommended use | Key parameters | Failure modes / notes |
|---|---|---:|---|---|
| `hfr` | HFRStarFocusMeasure | Fine tuning | `fwhm`, `star_find_threshold`, `cutout_size`, `max_stars`, `saturation_threshold`, `absolute_detection_limit` | Returns the half flux radius (HFR) in pixels. Requires reliable star detection; fails when there are too few stars, or when stars are too faint to be seen far from focus. |
| `gauss` | GaussianStarFocusMeasure | Fine tuning | same as `hfr` (Gaussian fit) | Returns the FWHM in pixels. Slower than HFR; sensitive to model mismatch (e.g. donuts) and saturated stars. |
| `fft` | FFTFocusMeasureTan2022 | Coarse search | none | Works on global frequency content; robust when stars are large and blurred; sensitive to large gradients or vignette and to image noise. |
| `fft_power` | FFTPowerFocusMeasure | Coarse search | none | Similar to `fft`; measures high-frequency power after noise floor subtraction. |
| `fft_phase_magnitude_product` | FFTPhaseMagnitudeProductFocusMeasure | Coarse search | none | Uses phase×magnitude; can be noisy for low SNR images. |
| `normalized_variance` | NormalizedVarianceFocusMeasure | Coarse search | none | Depends on mean image brightness; can be unstable with very low or uneven illumination. |
| `brenner` | BrennerFocusMeasure | Coarse search | none | Fast and simple gradient metric; noisy on low-SNR frames. |
| `tenengrad` | TenengradFocusMeasure | Coarse search (or fine if needed) | `ksize` (Sobel kernel size) | Good edge-sensitive measure; tune `ksize`; sensitive to hot pixels and noise. |
| `laplacian` | LaplacianFocusMeasure | Coarse search | none | Edge-based; can be affected by image noise and cosmic rays. |
| `variance_laplacian` | VarianceOfLaplacianFocusMeasure | Coarse search | none | Uses variance of Laplacian — robust to scale but noisy on low-SNR. |
| `absolute_gradient` | AbsoluteGradientFocusMeasure | Coarse search | none | Simple gradient sum; fast but less discriminative. |
| `squared_gradient` | SquaredGradientFocusMeasure | Coarse search | none | Similar to absolute gradient, emphasizes large gradients. |
| `auto_correlation` | AutoCorrelationFocusMeasure | Coarse search | none | Measures autocorrelation structure; can be sensitive to patterns or bright extended objects. |

Notes

- Parameters for star-size operators: `fwhm` and `star_find_threshold` control the star finder initialisation. `cutout_size` is the start half-width of the window per star; the window grows up to `8 * cutout_size` when a defocused star does not fit. `max_stars` limits runtime.
- Star-size operators subtract the sky (per row and per column, which also removes row banding), replace hot pixels, and use only pixels with star light. The focus measure is the median over stars. Sources whose size does not follow the focus over a sweep (e.g. hot pixels and noise) are not used. `hfr` returns the half flux radius and `gauss` the FWHM, in pixels.
- Sweeping autofocusers fail, and move the focuser back to its start position, if the focus curve is flat (see `min_focus_curve_snr`).
- Use a two-stage pipeline: start with a coarse operator (`fft`, `tenengrad`, `brenner`) across a wide range and coarse step size; switch to a star-size operator (`hfr`, `gauss`) for fine sweeps near the peak.
- To inspect available operators programmatically:

```python
from astrafocus import FocusMeasureOperatorRegistry
print(FocusMeasureOperatorRegistry.list())
```
