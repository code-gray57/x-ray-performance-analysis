
"""
This module contains a class and functions for MTF (Modulation Transfer Function) calculations.
The class is used simply for saving MTF data - specifically for the x-ray image performance analysis -
and the functions calculate the MTF for different things and different circumstances.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon

class MTF:
    """ This class is used to hold Modulation Transfer Function data for other objects.
    It does not contain any actual calculation methods - it is solely for storage. The other
    module functions are used for calculations.
    """

    def __init__(self):

        # For Spatial Resolution test, which is shared between N4255 and F792
        # Attributes for MTF data
        self.MTF20_x = None     # NOTE: Originally, MTF20_x was given 0.0 for "...image", but None in "...data"
        self.MTF20_y = None     # NOTE: Originally, MTF20_y was given 0.0 for "...image", but None in "...data"

        self.MTF_x = None
        self.MTF_x_f = None
        self.MTF_y = None
        self.MTF_y_f = None

        # Test_Object_Data specific attributes - will not be used in Test_Object_Image instances
        self.MTF20_x_arr = []
        self.MTF20_y_arr = []
        self.MTF20_y_err = None
        self.MTF20_x_err = None



#=============================================================#
# Here are the more general functions that work with MTF data #
#=============================================================#

def get_MTF_data(MTF, f):
    """ Calculate the MTF (Modulation Transfer Function) and return the MTF and frequency data

    :param MTF:
    :param f:
    :return
    """

    f_step = 0.02
    f_out = f_step * np.arange(int(np.max(f) / f_step) + 1)
    MTF_out = np.interp(f_out, f, MTF)

    return f_out, MTF_out


def determine_MTF_20(MTF, cy_per_mm, plot=False, plot_type=0):
    """ ADD GENERAL DESCRIPTION OF FUNCTION

    :param MTF:
    :param cy_per_mm:
    :param plot: flag for plotting the MTF20 data results
    :param plot_type:
    :return:
    """
    vals = np.where(MTF < 0.2, cy_per_mm, cy_per_mm[-1])
    MTF_20 = np.min(vals)

    # ADD COMMENT HERE
    if plot and plot_type == 4:
        plt.plot(cy_per_mm, MTF, 'ro')
        plt.plot([MTF_20, MTF_20], [0.1, 0.3])
        plt.plot([MTF_20 * 0.75, MTF_20 * 1.33], [0.2, 0.2])
        plt.xlabel('cycles per mm')

    return MTF_20


#-----------------------------------------------------------------------#
# This function calculates the LSF and MTF closely following the        #
# methods outlined in the ISO 12233:2000 standard. The one major        #
# difference between this code and the ISO method is that I detect      #
# the edge position using using the Radon transform here, since I think #
# that should be more reliable.                                         #
#-----------------------------------------------------------------------#
def slanted_edge_MTF(img, Radon=True):
    """ Calculate the LSF and MTF of the passed image.

    :param img: The image to calculate the LSF and MTF
    :param Radon: Flag to compute the Radon transform or not
    :return: the binned MTF data and the cycles per mm
    """

    # We use the Radon transform to find the angle of the brightest line
    # in the derivative img
    deriv_matrix = np.asarray([[-0.5, 0, 0.5]], np.float32)
    scale = 100.0 / np.max(img.shape)
    img_sm = img

    # comment here?
    if scale < 1.0:
        img_sm = cv2.resize(img_sm, (0, 0), fx=scale, fy=scale)
    LSF_img = cv2.filter2D(img_sm, -1, deriv_matrix)  # line spread function

    if Radon:
        # Step 1: course angle survey
        theta_array0 = np.linspace(-20, 20., 20, endpoint=False)
        sinogram0 = radon(LSF_img, theta=theta_array0, circle=False)
        ind0 = np.unravel_index(sinogram0.argmax(), sinogram0.shape)
        edge_theta0 = theta_array0[ind0[1]]

        # Step 2: medium angle survey
        theta_array1 = edge_theta0 + np.linspace(-2, 2., 11, endpoint=True)
        sinogram1 = radon(LSF_img, theta=theta_array1, circle=False)
        ind1 = np.unravel_index(sinogram1.argmax(), sinogram1.shape)
        edge_theta1 = theta_array1[ind1[1]]

        # Step 3: fine angle survey
        theta_array2 = edge_theta1 + np.linspace(-0.3, 0.3, 7, endpoint=True)
        sinogram2 = radon(LSF_img, theta=theta_array2, circle=False)
        ind2 = np.unravel_index(sinogram2.argmax(), sinogram2.shape)
        edge_theta = theta_array2[ind2[1]]

        shift_per_row = np.sin(edge_theta * 0.01745329)

    # ADD COMMENT HERE
    else:
        h = img.shape[0]
        grad_horiz = np.gradient(img)[1]

        max_arr1 = np.argmax(grad_horiz, axis=1)

        shift_per_row = np.mean(np.gradient(max_arr1))

    rho_vals = np.zeros(len(np.ravel(img)))  # this is rho as defined in the Radon transform

    n_cols = img.shape[1]
    n_rows = img.shape[0]

    # ADD COMMENT HERE
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            k = j + i * n_cols
            rho_vals[k] = (j * 1.0 - i * 1.0 * shift_per_row)

    # comment here?
    kernel = np.array([[0.0, 0.0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    img_lsf = cv2.filter2D(img, -1, kernel)

    # Following ISO 12233:2000, we bin into quarter pixels
    X4_binned = np.rint(rho_vals * 4)  # multimply by four then round -> X4
    bins = np.arange(np.min(X4_binned), np.max(X4_binned) + 1, 1)

    total_in_bin, void = np.histogram(X4_binned, bins, weights=np.ravel(img_lsf))
    n_per_bin, void = np.histogram(X4_binned, bins)

    LSF_binned = np.array(total_in_bin / n_per_bin)
    LSF_binned = LSF_binned[n_per_bin > 0]

    # Trim the data to n_rows/2 pts on either side of the max
    ind = np.argmax((LSF_binned))
    shift = ind - len(LSF_binned) / 2

    #
    if shift > 0:
        LSF_binned = LSF_binned[int(2 * shift):]
    else:
        LSF_binned = LSF_binned[0:int(2 * shift)]

    # ADD COMMENT HERE
    window = 0.54 - 0.46 * np.cos(2 * np.pi * (np.arange(len(LSF_binned))) / (len(LSF_binned) - 1))

    LSF_binned = LSF_binned * window
    MTF_binned = np.fft.fft(LSF_binned)
    MTF_binned = np.abs(MTF_binned / MTF_binned[0])

    # ADD COMMENT HERE
    cy_per_px = np.arange(len(MTF_binned)) * 4.0 / len(MTF_binned)
    cy_per_px = cy_per_px[0:int(len(MTF_binned) / 4.0 - 1)]
    MTF_binned = 1.0 * MTF_binned[0:int(len(MTF_binned) / 4.0 - 1)]

    return MTF_binned, cy_per_px


def MTF_from_slanted_sq(img_slanted_sq, ROI_size_px):
    """ ADD GENERAL DESCRIPTION OF FUNCTION

    :param img_slanted_sq:
    :param ROI_size_px:
    :return:
    """

    # Make sure the image argument holds the right data type
    img_slanted_sq = img_slanted_sq.astype(np.float32)
    img_size = img_slanted_sq.shape

    # ADD COMMENT HERE
    horiz_ROI = img_slanted_sq[
                int(img_size[0] / 2.0 - ROI_size_px[0] / 2):int(img_size[0] / 2.0 + ROI_size_px[0] / 2), :]

    horiz_left = horiz_ROI[int(horiz_ROI.shape[0] / 4):int(horiz_ROI.shape[0] * 3 / 4),
                           0:int(horiz_ROI.shape[1] / 2)]

    horiz_right = horiz_ROI[int(horiz_ROI.shape[0] / 4):int(horiz_ROI.shape[0] * 3 / 4),
                            int(horiz_ROI.shape[1] / 2):]

    # ADD COMMENT HERE
    vert_ROI = img_slanted_sq[:,
                int(img_size[1] / 2.0 - ROI_size_px[1] / 2):int(img_size[1] / 2.0 + ROI_size_px[1] / 2)]
    vert_top = vert_ROI[0:int(vert_ROI.shape[0] / 2), int(vert_ROI.shape[1] / 4):int(vert_ROI.shape[1] * 3 / 4)]
    vert_bottom = vert_ROI[int(vert_ROI.shape[0] / 2):, int(vert_ROI.shape[1] / 4):int(vert_ROI.shape[1] * 3 / 4)]

    # all these ROIs will be converted to to same orientation as horiz_right
    # so that they can be passed through the same analysis steps
    vert_bottom = np.rot90(vert_bottom, k=1)
    horiz_left = np.rot90(horiz_left, k=2)
    vert_top = np.rot90(vert_top, k=3)

    # Calculate the MTF and corresponding frequency grid for each ROI using the slanted_edge function/algorithm
    (MTF_horiz_right, f1) = slanted_edge_MTF(horiz_right)
    (MTF_horiz_left, f2) = slanted_edge_MTF(horiz_left)

    (MTF_vert_top, f3) = slanted_edge_MTF(vert_top)
    (MTF_vert_bottom, f4) = slanted_edge_MTF(vert_bottom)

    # Combine the MTF and frequency grid data into two distinct tuples and return them.
    MTFs = (MTF_horiz_right, MTF_vert_bottom, MTF_horiz_left, MTF_vert_top)
    frequency_grids = (f1, f4, f2, f3)

    return MTFs, frequency_grids