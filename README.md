# Telescope autofocuser

## Overview

The `telescope autofocuser` is a package designed to automate the autofocus process for telescopes. It comprises two main components:
1. targeting a specific section around the zenith
2. performing autofocus

## Example Usage

The file, `exploration/speculoos_main.py` provides an example of using the Autofocus Telescope System.
The key components include:
- The initialisation of the interface with the hardware
  - `TelescopeSpecs`: Loading specs of the telescope from a `.yaml` file (see e.g. `exploration/speculoos.yaml`)
  - `Telescope`: Encapsulating the interface of the telescope.
    - `TelescopePointer`: Interface for pointing the telescope to a specific coordinate in the equatorial coordinate system.
    - `TelescopeFocuser`: Interface for changing the focus position of the telescope.
- Targeting
  - `ZenithNeighbourhoodQuery`: Queries the zenith neighbourhood in a database to find a suitable section of the sky for focusing.
- Focsuing
  - `SweepingAutofocuser`: Performs the autofocusing using sweeping through a range of focus positions.
  - `AnalyticResponseAutofocuser(SweepingAutofocuser)`: Performs the autofocusing utilising the analytic nature of the focus response curve of a given focus measure.

For detailed usage and customization, refer to the source code and docstrings in each module.

## Installation

To install the package, clone the project and run:
```bash
python3 -m pip install
```
to install from source or
```bash
python -m pip install -e .
```
to install in editable mode.
For more information, consult the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-a-local-src-tree).

### <a name="catalogue"></a>The Gaia-2MASS Local Catalogue
The targeting procedure requires the `Gaia-2MASS Local Catalogue` which can be downlaoded [here](https://github.com/ppp-one/gaia-tmass-sqlite).

## Optional Dependencies
The package supports additional features through optional dependencies.
You can install these dependencies based on your needs. Choose from the following options:
```bash
# To also install visualization tools, including matplotlib, plotly and dash
python3 -m pip install .[visualization]
# To install packages for more statistics and machine learning, including scikit-learn.
python3 -m pip install .[extended]
# To install alpyca
python3 -m pip install .[alpaca]
```
`Alpyca`` is a [Python 3.7+ API library](https://pypi.org/project/alpyca/)
for all Astronomy Common Object Model ([ASCOM](https://ascom-standards.org/))
Alpaca universal interfaces.
This library provides a possible API for communication between this package and the devices required
for focussing, namely the camera and the focuser.

## Project Structure
The project structure includes several key directories:
- `autofocus`: The main package containing autofocus-related modules.
- `interface`: Subpackage with modules for interfacing with telescope components.
- `models`: Modules defining mathematical models used by some of the autofocus procedures.
- `sql`: Modules handling database queries of the Gaia-2MASS Local Catalogue (see [above](#catalogue)).
- `targeting`: Modules related to targeting specific regions in the sky.
- `utils`: General utility modules.

May your stars align and your focus be as sharp as a caffeinated owl spotting its prey!
