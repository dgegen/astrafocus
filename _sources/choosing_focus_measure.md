# Choosing a focus measure

When running an autofocus sweep you typically need two complementary strategies:

- **Coarse Search (e.g., `fft`)**: Use image-sharpness or global metrics when searching a wide focuser range. These non-parametric operators don't rely on detecting individual stars — they measure overall frame contrast or frequency content. They work well when stars are large and blurred ("donuts"), and are robust against optical distortions.

- **Fine Tuning (e.g., `HFR`)**: Use star-size estimators once you are near the focus peak. These detectors measure per-star diameters (Half Flux Radius / HFR) and let you fit a sharp "V-curve" for precise focus. They give high accuracy for point-like stars but will fail during broad searches if stars are too bloated to be reliably detected.

Practical recipe:

- Start with a coarse operator (FFT, Tenengrad) over a large range and coarse step size to find the approximate peak.
- Switch to a star-based operator (HFR, Gaussian fit) with finer steps around the peak to converge precisely.

This two-stage approach balances robustness across wide ranges with high precision near the optimum.
