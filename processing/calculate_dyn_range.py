"""
This module contains the function 'calculate_dyn_range()', which calculates the dynamic range (i.e. the brightest
and darkest pixel values) in a given x-ray image - specifically, it creates ROI for the brightest and darkest
areas.
This is used as a test in both the ANSI N42.55 and ASTM F792-OE standards.
"""

import cv2
import numpy as np

from utility.ROI import ROI


def calculate_dyn_range(full_img, step_wedge_ROI, boundaries, ROI_size, export_ROIs=False):
    """ ADD GENERAL DESCRIPTION OF FUNCTION HERE

     :param full_img: the complete image of the test object
     :param step_wedge_ROI: the ROI of just the step wedge test on the test object
     :param boundaries:
     :param ROI_size:
     :param export_ROIs: flag to indicate whether or not the dark and bright ROIs should be returned as well
     :return: The dynamic range of the image, as well as the ROIs of the dark and bright areas in the image if
             export_ROIs is True
     """

    # ADD COMMENT
    left_img = full_img[int(full_img.shape[0] * 1 / 10):int(full_img.shape[0] * 9 / 10),
               int(full_img.shape[1] / 10):int(full_img.shape[1] * 4 / 10)]

    left_img = cv2.blur(left_img, ((13, 13)), borderType=cv2.BORDER_REFLECT)
    bright_val = np.max(left_img)
    bright_ind = np.unravel_index(left_img.argmax(), left_img.shape)

    # ADD COMMENT
    ROIs = []

    center1 = step_wedge_ROI.center[1]
    center0_offset = step_wedge_ROI.center[0] - step_wedge_ROI.shape[0] / 2

    # ADD COMMENT
    for i in range(len(boundaries) - 1):
        center0 = (boundaries[i] + boundaries[i + 1]) / 2 + center0_offset
        ROIs.append(ROI(full_img, (center0, center1), ROI_size))

    stdev_arr = np.zeros(len(ROIs))
    mean_arr = np.zeros(len(ROIs))

    print(len(ROIs))

    # ADD COMMENT
    for i, ROI_i in enumerate(ROIs):
        stdev_arr[i] = ROI_i.get_stdev()
        mean_arr[i] = ROI_i.get_ave()
        ROI_i.add_rect_to_plot()

    min_ind = mean_arr.argmin()
    stdev_min = stdev_arr[min_ind]

    # ADD COMMENT
    if stdev_min < 1.0:
        # s
        stdev_arr = stdev_arr[stdev_arr > 1.0]

        # Small comment?
        if len(stdev_arr) == 0:
            stdev_arr = np.nan
        else:
            stdev_min = np.min(stdev_arr)

    dynamic_range = bright_val / stdev_min


    # If export_ROIs is true, then create ROIs for the dark and bright
    # parts of the image and return them as well
    if export_ROIs:
        center = np.array(full_img.shape) / 10 + bright_ind
        bright_ROI = ROI(full_img, center, (5, 5))
        dark_ROI = ROIs[min_ind]

        return dynamic_range, dark_ROI, bright_ROI

    # Otherwise, return just the dynamic range
    return dynamic_range


