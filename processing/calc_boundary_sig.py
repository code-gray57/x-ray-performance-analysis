
"""
This module contains the 'calc_boundary_sig()' function, which calculates the boundary signal within a given
ROI of an x-ray image. It is used in the Organic BSNR test for both ANSI N42.55 and ASTM F792-OE standards.
"""

import numpy as np
import matplotlib.pyplot as plt

from utility.ROI import ROI


def calc_boundary_sig(ROI_in, px_size, ROI_size_mm=(2.0, 2.0), plot=False, force_middle=False):
    """ ADD GENERAL OVERVIEW OF FUNCTION

    :param ROI_in: The ROI that dictates where to calculate the boundary signal
    :param px_size: the pixel size for the image
    :param ROI_size_mm: The actual size of the ROI in millimeters
    :param plot: Flag to plot the ROI image and the two ROIs of the high and low values
    :param force_middle: Flag to force the y boundary to the middle of the image
    :return: The boundary signal of the ROI
    """

    # Get the image that the ROI dictates and find the row_sum
    img = ROI_in.get_img()
    row_sum = img.sum(axis=1)

    # -----------------------------------------------------------------------------------#
    # Input validation:                                                                 #
    #     Check that there are both BSNR image data as well as a pixel size for the ROI #
    # -----------------------------------------------------------------------------------#
    if len(row_sum) == 0:
        print('BSNR img empty... ', ROI_in.name)
        return 1e-6

    if px_size is None:
        print('This shouldnt happen' + a + b)
        print('Pixel size is required for measuring the boundary signal (is not None).')
        return 1e-6  # Exit with error code instead??

    # Calculate the ROI pixel size
    ROI_size_px = np.array(ROI_size_mm) / np.array(px_size)

    # Smooth the rowsum then take the derivative
    row_sum = np.convolve(np.ones(3) / 3, row_sum, 'same')
    deriv = np.gradient(row_sum)

    # Zero the edges of the deriv array because they are affected by artifacts
    n_ignore = 5  # ignore 1/n of the edge
    deriv[0:int(len(deriv) / n_ignore)] = 0
    deriv[int(-len(deriv) / n_ignore):] = 0

    # The maximum derivative amplitude happens at the boundary
    y_boundary = np.argmax(np.abs(deriv))

    # If y_boundary isn't near the middle then the signal must be completely flat
    # therefore, set it in the middle of the image
    if y_boundary < len(deriv) / n_ignore:
        y_boundary = len(deriv) / 2.0
    if y_boundary > (n_ignore - 1) * len(deriv) / n_ignore:
        y_boundary = len(deriv) / 2.0
    if force_middle:
        y_boundary = len(deriv) / 2.0

    # --------------------------------------------------------------------#
    # Create the two ROI's based upon the y boundary and the image shape #
    # --------------------------------------------------------------------#

    # ROI 1
    center1 = (int(y_boundary + 1.5 * ROI_size_px[1]), int(img.shape[1] / 2))
    ROI_lo = ROI(img, center1, ROI_size_px)

    # ROI 2
    center2 = (int(y_boundary - 1.5 * ROI_size_px[1]), int(img.shape[1] / 2))
    ROI_hi = ROI(img, center2, ROI_size_px)

    # Add the sub-ROIs to the list of the main ROI
    ROI_in.add_subROI(ROI_hi)
    ROI_in.add_subROI(ROI_lo)

    # Plot the two new ROI's
    if plot:
        plt.imshow(img, cmap=plt.get_cmap('gray'))  # ,vmin=vmin,vmax=vmax)
        plt.autoscale(False)

        ROI_lo.add_rect_to_plot()
        ROI_hi.add_rect_to_plot()

    # Calculate the average of the two ROIs and then the boundary signal
    hi_ROI_ave = ROI_hi.get_ave()
    low_ROI_ave = ROI_lo.get_ave()

    boundary_signal = 1.0 - low_ROI_ave / hi_ROI_ave

    # Return the boundary signal.
    return boundary_signal