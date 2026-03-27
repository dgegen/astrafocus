from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from astrafocus.focus_measure_operators import FocusMeasureOperator


class FocusMeasureScan:
    """Run one or more focus measure operators over a sequence of FITS images.

    Parameters
    ----------
    operators : list[FocusMeasureOperator]
        One or more focus measure operator instances to apply to each image.

    Examples
    --------
    >>> from astrafocus import FocusMeasureOperatorRegistry
    >>> from astrafocus.focus_measure_scan import FocusMeasureScan
    >>> scan = FocusMeasureScan.from_names(["Brenner", "Tenengrad", "FFT"])
    >>> scan = FocusMeasureScan.from_names(["FFT"])
    >>> scan.operators.append(FocusMeasureOperatorRegistry.from_name("HFR")(fwhm=4.0))
    """

    def __init__(self, operators: list[FocusMeasureOperator]):
        self.operators = operators

    @classmethod
    def from_names(cls, names: list[str]) -> "FocusMeasureScan":
        """Construct a FocusMeasureScan from a list of operator name strings.

        Parameters
        ----------
        names : list[str]
            Operator names as accepted by ``FocusMeasureOperatorRegistry.from_name``
            (e.g. ``["brenner", "tenengrad"]``).
        """
        from astrafocus import FocusMeasureOperatorRegistry

        return cls([FocusMeasureOperatorRegistry.from_name(name)() for name in names])

    def run(
        self,
        directory: str | Path,
        pattern: str = "*.fits",
        header_fields: list[str] | None = ["DATE-OBS", "FOCUSPOS"],
    ) -> pd.DataFrame:
        """Compute focus measures for all FITS files in a directory.

        Parameters
        ----------
        directory : str or Path
            Directory containing FITS files.
        pattern : str
            Glob pattern used to find FITS files. Default is ``"*.fits"``.
        header_fields : list[str], optional
            FITS header keywords to extract into the result DataFrame (e.g. ``["FOCPOS"]``).

        Returns
        -------
        pd.DataFrame
            One row per file. Columns: ``file``, any requested ``header_fields``,
            and one column per operator named by ``operator.name``.

        Examples
        --------
        >>> import tempfile
        >>> import cabaret
        >>> from cabaret.sources import Sources
        >>> import numpy as np
        >>> from astrafocus.focus_measure_scan import FocusMeasureScan
        >>> sources = Sources.from_arrays(
        ...     ra=np.array([10.684]), dec=np.array([41.269]), fluxes=np.array([1e5])
        ... )
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     obs = cabaret.Observatory(focuser={"position": 10000, "best_position": 10000})
        ...     _ = obs.generate_fits_image(
        ...         ra=10.684, dec=41.269, sources=sources, exp_time=10, seed=42,
        ...         file_path=f"{tmpdir}/focus_10000.fits",
        ...         user_header={"FOCUSPOS": 10000},
        ...     )
        ...     scan = FocusMeasureScan.from_names(["Brenner"])
        ...     df = scan.run(tmpdir, header_fields=["FOCUSPOS"])
        ...     list(df.columns)
        ['file', 'FOCUSPOS', 'Brenner']
        """
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")

        rows = []
        with tqdm(paths, desc="Scanning") as progress:
            for path in progress:
                progress.set_postfix_str(path.name)
                with fits.open(path) as hdul:
                    image = hdul[0].data.astype(np.float64)  # type: ignore
                    header = hdul[0].header  # type: ignore

                row: dict = {"file": path.name}
                for field in header_fields or []:
                    row[field] = header.get(field)
                for op in self.operators:
                    try:
                        row[op.name] = op(image)
                    except Exception:
                        row[op.name] = np.nan
                rows.append(row)

        return pd.DataFrame(rows)

    def plot_all(self, df, plot_kwargs={}, log_scale=False, axes=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting focus measures")

        plot_kwargs = {"marker": "o", "ls": "", "color": "black"} | plot_kwargs

        operators = [op for op in self.operators if op.name in df.columns]

        if axes is None:
            _, axes = plt.subplots(len(operators), 1, figsize=(10, 5 * len(operators)), sharex=True)
            axes = axes if len(operators) > 1 else [axes]

        # Plot every operator in a separate subplot
        # Sort wrt. to focus position if available, otherwise by file name

        if "FOCUSPOS" in df.columns:
            df = df.sort_values("FOCUSPOS")
            x = df["FOCUSPOS"]
            x_label = "Focus Position"
        else:
            df = df.sort_values("file")
            x = range(len(df))
            x_label = "File Index"

        for ax, op in zip(axes, operators):
            ax.plot(x, df[op.name], **plot_kwargs)
            direction = "$\\downarrow$ better" if op.smaller_is_better else "$\\uparrow$ better"
            ax.set_ylabel(f"{op.name}\n{direction}")
            if log_scale:
                ax.set_yscale("log")

        axes[-1].set_xlabel(x_label)

        return axes
