"""
This module contains the function 'calculate_img_extent()', which finds the extent of a given image - normally an image
specifically added for finding the extent. It is used as a part of the ANSI N42.55 standard tests, but not in the
ASTM F792 tests.
"""

import cv2
import numpy as np

from utility.ROI import ROI


def calculate_img_extent(img, px_size, ROI_size_mm=1.0):
    """ ADD GENERAL DESCRIPTION OF FUNCTION

    :param img - The image to calculate the extend from
    :param px_size - The pixel size in the image (in millimeters?)
    :param ROI_size_mm - The actual size of the ROI in millimeters
    :return: The processed image extent, two ROIs of the dark and bright image areas, and the dynamic range
    """

    # If the image is in color, compress it to greyscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ADD COMMENT
    n_blur = 40
    img_blurred = cv2.blur(img, (n_blur, n_blur), borderType=cv2.BORDER_REFLECT)

    bright_ind = np.unravel_index(img_blurred.argmax(), img_blurred.shape)
    bright_ROI = ROI(img, bright_ind, (10, 10))
    bright_val = np.max(img_blurred)

    # ADD COMMENT
    distances = np.array([1.0, 5.0, 10.0])  # in mm
    dyn_range = np.array(distances) * 0
    dark_ROIs = []

    # ADD COMMENT
    for i, distance in enumerate(distances):
        d_px = np.round(distance / px_size[0])
        ROI_size_px = np.round(ROI_size_mm / np.array(px_size))
        dark_roi_blurred = img_blurred[int(0 - d_px - ROI_size_px[0] / 2): int(0 - d_px + ROI_size_px[0] / 2), :]

        # ADD COMMENT
        dark_ind0 = np.unravel_index(dark_roi_blurred.argmin(), dark_roi_blurred.shape)
        dark_indi = (img.shape[0] - d_px, dark_ind0[1])  # in coords of orig img

        dark_ROIi = ROI(img, dark_indi, ROI_size_px)
        dark_ROIs.append(dark_ROIi)
        stdev = dark_ROIi.get_stdev()

        # Comment here?
        dynamic_range = bright_val / stdev
        dyn_range[i] = dynamic_range

    # ADD COMMENT
    if np.max(dyn_range) >= 150:
        extent = np.min(distances[dyn_range > 150])
        ind = np.where(dyn_range > 150)
        ind = np.min(ind)
        dark_ROI = dark_ROIs[ind]

    # Otherwise, the extent could not be found and
    else:
        extent = None
        dark_ROI = dark_ROIs[-1]

    return extent, dark_ROI, bright_ROI, dyn_range


