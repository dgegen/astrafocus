from pathlib import Path

import pandas as pd
import pytest

from astrafocus.focus_measure_scan import FocusMeasureScan


def test_run_returns_dataframe(fits_directory: Path):
    scan = FocusMeasureScan.from_names(["Brenner", "Tenengrad"])
    df = scan.run(fits_directory, header_fields=["FOCUSPOS"])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["file", "FOCUSPOS", "Brenner", "Tenengrad"]
    assert len(df) == 5
    assert df["FOCUSPOS"].tolist() == sorted(df["FOCUSPOS"].tolist())


def test_run_no_header_fields(fits_directory: Path):
    scan = FocusMeasureScan.from_names(["Brenner"])
    df = scan.run(fits_directory, header_fields=None)

    assert "FOCUSPOS" not in df.columns
    assert "Brenner" in df.columns


def test_run_missing_directory():
    scan = FocusMeasureScan.from_names(["Brenner"])
    with pytest.raises(FileNotFoundError):
        scan.run("/nonexistent/path")


def test_plot_all_returns_axes(fits_directory: Path):
    pytest.importorskip("matplotlib")
    scan = FocusMeasureScan.from_names(["Brenner", "Tenengrad"])
    df = scan.run(fits_directory, header_fields=["FOCUSPOS"])
    axes = scan.plot_all(df)
    assert len(axes) == 2
