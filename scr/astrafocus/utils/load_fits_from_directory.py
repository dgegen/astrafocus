import glob
import os

import numpy as np
from astropy.io import fits


def load_fits_from_directory(directory):
    # Use glob to get a list of FITS files in the directory
    fits_files = np.array(glob.glob(os.path.join(directory, "*.fit")))

    headers = []
    for file_name in fits_files:
        with fits.open(file_name) as hdul:
            headers.append(hdul[0].header)

    focus_pos = np.array([header["FOCUSPOS"] for header in headers])
    # sort_ind = np.argsort(focus_pos)
    sort_ind = np.argsort([header['JD'] for header in headers])
    

    headers = [headers[i] for i in sort_ind]
    focus_pos = focus_pos[sort_ind]

    image_data = []
    for file_name in fits_files[sort_ind]:
        with fits.open(file_name) as hdul:
            data = hdul[0].data
            image_data.append(data)

    return image_data, headers, focus_pos
