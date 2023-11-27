from typing import NamedTuple, Optional

import astropy.units as u
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import Angle, EarthLocation

from autofocus.targeting.airmass_models import find_airmass_threshold_crossover, plane_parallel_atmosphere


class RangeBase(NamedTuple):
    min: float
    max: float


class FocusRange(RangeBase):
    pass


class MagnitudeRange(RangeBase):
    pass


class PixelShape(NamedTuple):
    x: int
    y: int


class FieldOfView(NamedTuple):
    height: u.Quantity
    width: u.Quantity


class TelescopeSpecs:
    """
    Class representing the specifications of a telescope.

    Parameters
    ----------
    name : str
        Name of the telescope.
    fov_width : float
        Width of the field of view in arcminutes.
    fov_height : float
        Height of the field of view in arcminutes.
    pixel_shape : dict
        Dictionary containing 'x' and 'y' keys representing pixel dimensions.
    pixel_scale : float
        Pixel scale in arbitrary units.
    observatory_location : EarthLocation
        Location of the observatory specified using astropy's EarthLocation.
    focus_range : dict
        Dictionary containing 'min' and 'max' keys representing the focus range.
    max_airmass : float
        Maximum airmass allowed for observations.
    g_mag_range : dict
        Dictionary containing 'min' and 'max' keys representing the g-band magnitude range.
    j_mag_range : dict
        Dictionary containing 'min' and 'max' keys representing the J-band magnitude range.
    gaia_tmass_db_path : Optional[str], optional
        Path to the Gaia-Tycho-2 mass database (default is None).

    Examples
    --------
    telescope_specs = TelescopeSpecs.load_telescope_config(file_path="path_to/config.yaml")
    """

    def __init__(
        self,
        name: str,
        fov_width: float,
        fov_height: float,
        pixel_shape: dict,
        pixel_scale: float,
        observatory_location: EarthLocation,
        focus_range: dict,
        max_airmass: float,
        g_mag_range: dict,
        j_mag_range: dict,
        gaia_tmass_db_path: Optional[str],
    ):
        self.name = name
        self.fov = FieldOfView(width=fov_width, height=fov_height)
        self.pixel_shape = PixelShape(pixel_shape['x'], pixel_shape['y'])
        self.pixel_scale = pixel_scale
        self.observatory_location = observatory_location
        self.focus_range = FocusRange(min=focus_range["min"], max=focus_range["max"])
        self.max_airmass = max_airmass
        self.g_mag_range = MagnitudeRange(min=g_mag_range["min"], max=g_mag_range["max"])
        self.j_mag_range = MagnitudeRange(min=j_mag_range["min"], max=j_mag_range["max"])

        self.gaia_tmass_db_path = gaia_tmass_db_path

    @classmethod
    def load_telescope_config(cls, file_path):
        """
        Load telescope configuration from a YAML file.
        
        Parameters
        ----------
        file_path : str
            Path to the YAML configuration file.

        Returns
        -------
        TelescopeSpecs
            An instance of the TelescopeSpecs class initialized with the configuration.
        """        
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Extract data with units
        lat = config_data["telescope"]["coordinates"]["lat"] * u.deg
        lon = config_data["telescope"]["coordinates"]["lon"] * u.deg
        height = config_data["telescope"]["coordinates"]["height"] * u.m

        fov_width = config_data["telescope"]["field_of_view"]["width"] * u.arcmin
        fov_height = config_data["telescope"]["field_of_view"]["height"] * u.arcmin

        focus_range = config_data["telescope"]["focus_range"]

        max_airmass = config_data["telescope"]["max_airmass"]
        g_mag_range = config_data["telescope"]["g_mag_range"]
        j_mag_range = config_data["telescope"]["j_mag_range"]
        pixel_shape = config_data["telescope"]["pixel_shape"]
        pixel_scale = config_data["telescope"]["pixel_scale"]

        gaia_tmass_db_path = config_data.get("gaia_tmass_db_path", None)

        # Create and return an instance of the TelescopeConfig class
        return cls(
            name=config_data["telescope"]["name"],
            fov_width=fov_width,
            fov_height=fov_height,
            pixel_shape=pixel_shape,
            pixel_scale=pixel_scale,
            observatory_location=EarthLocation(lat=lat, lon=lon, height=height),
            focus_range=focus_range,
            max_airmass=max_airmass,
            gaia_tmass_db_path=gaia_tmass_db_path,
            g_mag_range=g_mag_range,
            j_mag_range=j_mag_range,
        )

    def to_dict(self):
        """Convert TelescopeSpecs object to a dictionary."""        
        return {
            "name": self.name,
            "fov": {
                "width": self.fov.width.value,
                "height": self.fov.height.value,
                "unit": str(self.fov.width.unit),
            },
            "pixel_shape": self.pixel_shape,
            "observatory_location": {
                "lat": self.observatory_location.lat.value,
                "lon": self.observatory_location.lon.value,
                "height": self.observatory_location.height.value,
                "unit": str(self.observatory_location.lat.unit),
            },
            "focus_range": self.focus_range,
            "max_airmass": self.max_airmass,
            "g_mag_range": self.g_mag_range,
            "j_mag_range": self.j_mag_range,
            "gaia_tmass_db_path": self.gaia_tmass_db_path,
        }

    def find_airmass_threshold_crossover(self, airmass_model=plane_parallel_atmosphere):
        """Find the maximum zenith angle corresponding to the specified airmass threshold.
        
        Parameters
        ----------
        airmass_model : Callable, optional
            A function representing the atmospheric model (default is plane_parallel_atmosphere).

        Returns
        -------
        Angle
            Maximum zenith angle in degrees.
        """        
        max_zenith_angle = find_airmass_threshold_crossover(
            airmass_threshold=self.max_airmass, airmass_model=airmass_model
        )
        return Angle(max_zenith_angle * 180 / np.pi * u.deg)

    def __repr__(self):
        return (
            f"TelescopeSpecs(name={self.name!r}, fov=FieldOfView(width={self.fov.width!r}, height={self.fov.height!r}), "
            f"pixel_shape={self.pixel_shape!r}, observatory_location={self.observatory_location!r}, "
            f"focus_range={self.focus_range!r}, max_airmass={self.max_airmass!r}, "
            f"g_mag_range={self.g_mag_range!r}, j_mag_range={self.j_mag_range!r}, "
            f"gaia_tmass_db_path={self.gaia_tmass_db_path!r})"
        )
