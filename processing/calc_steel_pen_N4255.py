"""
This module contains the function "calc_steel_pen_N4255()", which calculates the useful penetration of steel
by the x-ray system given an x-ray image of the test object.

Note that while both ANSI N42.55 and ASTM F792-OE both have a useful penetration test as part of the standards,
this function is specifically for calculating the results according to the N42.55 standard.
"""

import numpy as np
import matplotlib.pyplot as plt

from utility.ROI import ROI
from processing.calc_boundary_sig import calc_boundary_sig


def calc_steel_pen_N4255(img, ROI_size_px, px_size, plot=False, pdf_obj=None):
    """ Calculate the useful penetration of an x-ray system on the N42.55 step wedge.
    Go through each step in the step wedge and calculate the boundary signal between the
    two steps to determine whether or not the steps are distinguishable from each other
    in the x-ray image.

    :param img:
    :param ROI_size_px:
    :param px_size:
    :param plot: Flag for plotting the image and ROIs and displaying them OR saving them to a pdf
    :param pdf_obj: PdfPages object to save the figures (if plotted) to a pdf
    :return: An array for both the boundary positions and boundary signals
    """

    row_sum = img.sum(axis=1)

    # Smooth the rowsum then take the derivative
    N = 5
    row_sum_smoothed = np.convolve(row_sum, np.ones((N,)) / N, mode='valid')
    deriv = np.gradient(row_sum_smoothed)

    # -------------------------------------------------------------------------------#
    # Zero the edges of the derivative array because they are affected by artifacts #
    # -------------------------------------------------------------------------------#
    step_height_px = 15.0 / px_size[1]

    acorr = autocorr(deriv)
    acorr[0:int(step_height_px * 0.9)] = 0
    acorr[int(step_height_px * 1.1):] = 0
    step_height_px = np.argmax(acorr)

    deriv[int(2 * len(deriv) / 3):] = 0
    deriv[0:5] = 0

    # The maximum deriv amplitude happens at the 1st boundary
    y_boundary03 = np.argmin(deriv)

    # -------------------------------------------------#
    # Now we zero out everything within 2 ROI of this #
    # -------------------------------------------------#

    deriv[0:int(y_boundary03 + 2 * ROI_size_px[1])] = 0
    y_boundary36 = np.argmax(np.abs(deriv))

    deriv[0:int(y_boundary36 + 2 * ROI_size_px[1])] = 0
    y_boundary69 = np.argmax(np.abs(deriv))

    # ADD COMMENT HERE
    step_height_px_new6 = y_boundary69 - y_boundary36
    update_diff6 = np.abs(step_height_px_new6 - step_height_px) / step_height_px

    step_height_px_new3 = y_boundary36 - y_boundary03
    update_diff3 = np.abs(step_height_px_new3 - step_height_px) / step_height_px

    # ADD COMMENT HERE
    if update_diff6 < 0.2:
        step_height_px = step_height_px_new6 * 1.006
        boundary_pos_arr = y_boundary69 + (np.arange(13) - 2) * step_height_px
        boundary_pos_arr[0] = y_boundary03

    elif update_diff3 < 0.2:
        step_height_px = step_height_px_new3 * 1.02
        boundary_pos_arr = y_boundary36 + (np.arange(13) - 1) * step_height_px

    else:
        boundary_pos_arr = y_boundary03 + (np.arange(13)) * step_height_px

    # ADD COMMENT HERE
    if plot:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.autoscale(False)
        linewidth = 3

        # ADD COMMENT HERE
        for i, boundary_pos in enumerate(boundary_pos_arr):
            plt.plot((0, img.shape[1] - 1), (boundary_pos, boundary_pos), linewidth=linewidth)
            plt.text(img.shape[1] / 2, boundary_pos + step_height_px * 0.45, str((i + 1) * 3) + ' mm thick step',
                     horizontalalignment='center', color='white')

        if pdf_obj is None:
            plt.show()
        else:
            pdf_obj.savefig(fig)
        plt.close(fig)

    boundary_signal_arr = np.zeros(12)
    step_wedge_boundaries = np.zeros(13)

    if plot:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)

    # ADD COMMENT HERE
    for i in range(12):
        y_boundary_i = boundary_pos_arr[i]

        center = (y_boundary_i, img.shape[1] / 2 - 0.5)
        ROI_size_px = (step_height_px * 1.4, img.shape[1])

        # ADD COMMENT HERE
        if center[0] < 0 or center[0] > img.shape[0]:
            break

        ROI_vicinity = ROI(img, center, ROI_size_px)

        # ADD COMMENT HERE
        plt.subplot(4, 3, i + 1)
        bs = calc_boundary_sig(ROI_vicinity, px_size, ROI_size_mm=2, plot=plot, force_middle=True)
        boundary_signal_arr[i] = bs

        # Display the image of the main ROI and the areas of the subROIs
        if plot:
            plt.imshow(ROI_vicinity.get_img(), cmap=plt.get_cmap('gray'))
            subROIs = ROI_vicinity.subROIs
            for subROI in subROIs:
                subROI.add_rect_to_plot()

    # ADD COMMENT HERE - unclear exactly what is being plotted, since there's been so many so far.
    if plot:
        if pdf_obj is None:
            plt.show()
        else:
            pdf_obj.savefig(fig)
        plt.close(fig)

    return boundary_pos_arr, boundary_signal_arr


def autocorr(x):
    """ Compute the autocorrelation of the argument and returns the first half. """
    result = np.correlate(x, x, mode='full')

    return result[int(result.size / 2):]

