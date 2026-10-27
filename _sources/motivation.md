# Motivation

Good focus is a prerequisite for useful astronomical data. Even a small defocus broadens star profiles, reduces peak signal, and degrades the limiting magnitude of an exposure. In practice, focus drifts over the course of a night as the telescope structure responds to temperature changes, and it must be re-established whenever a new filter or instrument configuration is used.

Manual focusing is slow and operator-dependent. Existing autofocus tools are often tightly coupled to a single camera control application, making them hard to adapt to different hardware setups or custom observing workflows.

AstrAFocus addresses this by separating the autofocus logic from the hardware layer:

- **Device-agnostic** — the `AutofocusDeviceManager` abstraction wraps any camera and focuser; connect real hardware by implementing two small interface classes, or use the built-in simulators for development and testing.
- **Pluggable focus metrics** — 13 focus measure operators are included, from star-size estimators (HFR, Gaussian fit) to image-sharpness metrics (FFT, Tenengrad, Laplacian). Adding a new metric requires implementing a single method.
- **Flexible sweep strategies** — choose between an analytic curve fit (parabola) or non-parametric smoothing (LOWESS, spline, RBF) to locate the focus optimum.
- **Sky targeting** — a separate `ZenithNeighbourhoodQuery` component finds suitable focus fields near zenith from a local or remote Gaia-2MASS catalogue, so the telescope can slew to a well-populated region before starting the autofocus run.
