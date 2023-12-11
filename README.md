# AstrAFocus
AstrAFocus is a package that provides flexible autofocus (AF) procedures for telescopes.
The tasks that this package aims to automate fall into two broad categories.
1. **Locate the focus calibration region**: 
   - Identify an area around the zenith that is suitable for focus calibration, given the chosen
   focus measurement operator.
2. **Perform autofocus**: 
   - Select the focus positions at which exposures should be taken.
   - Measure the focus of an exposure at a given focus position using a focus measure operator.
   - Estimate the point of optimal focus given a set of exposures taken at different focus positions.
<p align="center">
   <img src="docs/logo.png" alt="logo image" width="45%" style="margin: 20px;">
</p>

## Example Usage

The file, `exploration/speculoos_main.py` provides an example of using the Autofocus Telescope System.
The key components include:
- The initialisation of the interface with the hardware
  - `TelescopeSpecs`: Loading specs of the telescope from a `.yaml` file (see e.g. `exploration/speculoos.yaml`)
  - `AutofocuserDeviceInterface`: The interface between the API of the devices and this library.
    - `CameraInterface`: Interface for performing exposures.
    - `FocuserInterface`: Interface for changing the focus position of the telescope.
    - `TelescopeInterface`: The implementation of this interface is optional, as the telescope
    only needs to be oriented once after the focus calibration region has been determined 
    with the targeting class `ZenithNeighbourhoodQuery`, which can be implemented externally.
    Accordingly, this swaying of the telescope can also be implemented externally. Note that it is
    currently assumed that the target tracking is controlled outside of this package.
- Targeting
  - `ZenithNeighbourhoodQuery`: Queries the zenith neighbourhood in a database to find a suitable section of the sky for focusing.
- Focsuing
  - `SweepingAutofocuser`: Performs the autofocusing using sweeping through a range of focus positions.
  - `AnalyticResponseAutofocuser(SweepingAutofocuser)`: Performs the autofocusing utilising the analytic nature of the focus response curve of a given focus measure.
  - `NonParametricResponseAutofocuser(SweepingAutofocuser)`: Performs the autofocusing using an arbitrary focus measure operator and then applying an extremum estimator from to find the position of maximal focus.


For detailed usage and customization, refer to the source code and docstrings in each module.

## Installation

To install the package, clone the project and run:
```bash
python3 -m pip install
```
to install from source or
```bash
python3 -m pip install -e .
```
to install in editable mode.
For more information, consult the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-a-local-src-tree).

### <a name="catalogue"></a>The Gaia-2MASS Local Catalogue
The targeting procedure requires the `Gaia-2MASS Local Catalogue` which can be downlaoded [here](https://github.com/ppp-one/gaia-tmass-sqlite).

### Optional Dependencies
The package supports additional features through optional dependencies.
You can install these dependencies based on your needs. Choose from the following options:
```bash
# To also install visualization tools, including matplotlib, plotly and dash
python3 -m pip install ".[visualization]"

# To install packages for more statistics and machine learning, including scikit-learn.
python3 -m pip install ".[extended]"

# To install dash
python3 -m pip install ".[dash]"

# To install alpyca
python3 -m pip install ".[alpaca]"
```
`Alpyca` is a [Python 3.7+ API library](https://pypi.org/project/alpyca/)
for all Astronomy Common Object Model ([ASCOM](https://ascom-standards.org/))
Alpaca universal interfaces.
This library is a possible API for communication between this package and the devices required
for focussing, namely the camera and the focuser.

## Project Structure
The project structure includes several key directories:
- `astrafocus`: The main package containing autofocus-related modules.
   - `interface`: Subpackage with modules for interfacing with the devices through their API components.
   - `models`: Modules defining mathematical models used by some of the autofocus procedures.
   - `sql`: Modules handling database queries of the Gaia-2MASS Local Catalogue (see [above](#catalogue)).
   - `targeting`: Modules related to targeting specific regions in the sky.
   - `utils`: General utility modules.

May your stars align and your focus be as sharp as a caffeinated owl spotting its prey!
