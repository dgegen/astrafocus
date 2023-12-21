import glob
import os

import numpy as np
from astropy.io import fits


def load_fits_with_focus_pos_from_directory(
    directory, suffix=".fit", focus_pos_key="FOCUSPOS", time_key="JD"
):
    image_data, headers = load_fits_from_directory(directory, suffix=suffix, sort_key=time_key)
    focus_pos = np.array([header[focus_pos_key] for header in headers])

    return image_data, headers, focus_pos


def load_fits_from_directory(directory, suffix=".fits", sort_key=None):
    """
    Load all fits files from a directory and return the image data and headers.

    By default, the order of the fits files is alphabetical by file name, assuming
    the file names are sorted by time of observation, as is often the case.

    Parameters
    ----------
    directory : str
        Path to directory containing fits files.
    suffix : str, optional
        Suffix of fits files to load. The default is ".fits".
    sort_key : str, optional
        Key to sort headers by. The default is None.

    Returns
    -------
    image_data : list
        List of image data from fits files.
    headers : list
        List of headers from fits files.
    """
    fits_files = np.sort(np.array(glob.glob(os.path.join(directory, f"*{suffix}"))))

    headers = []
    for file_name in fits_files:
        with fits.open(file_name) as hdul:
            headers.append(hdul[0].header)

    if sort_key is not None:
        sort_ind = np.argsort([header[sort_key] for header in headers])
        headers = [headers[i] for i in sort_ind]
        fits_files = fits_files[sort_ind]

    image_data = []
    for file_name in fits_files:
        with fits.open(file_name) as hdul:
            data = hdul[0].data
            image_data.append(data)

    return image_data, headers
