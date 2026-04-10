# *AstrAFocus*

A Python package for automated telescope focusing.

AstrAFocus sweeps a focuser through a range of positions, measures focus quality at each step, and determines the optimal position — all through a hardware-agnostic interface that works with real equipment or built-in simulators.

## Key features

- Two autofocus strategies: analytic curve fit (parabola) and non-parametric smoothing (LOWESS, spline, RBF)
- 13 focus measure operators: star-size estimators (HFR, Gaussian) and image-sharpness metrics (FFT, Tenengrad, Laplacian, …)
- Device abstraction: plug in any camera and focuser, or use the `CabaretDeviceSimulator` for hardware-free development
- Sky targeting: find suitable focus fields near zenith from a Gaia-2MASS catalogue

---

```{toctree}
   :maxdepth: 1
   :caption: Introduction

installation.md
contributing.md
motivation.md
notebooks/getting_started.ipynb
   choosing_focus_measure.md
   operators_reference.md
```

```{toctree}
   :maxdepth: 1
   :caption: References

api.md
citation.md
```
