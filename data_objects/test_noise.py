
"""
This module contains classes that represent blank/noise x-ray images to be used for analysis.

It contains 3 classes, one superclass that contains the majority of the methods and attributes, with
two subclasses that cater specifically to either the ANSI N42.55 or ASTM F792 standard for some
of the attributes.
"""

import cv2
import numpy as np

from utility.ROI import ROI


class Test_Object_Noise:
    """ This class represents a blank / noise x-ray image, used for testing the x-ray system
    that took the image against a standard for analysis of the system's capabilities. """

    def __init__(self, img_in, px_size, filename=None, name='noise'):

        # If the passed image is in color, compress it into greyscale
        if len(img_in.shape) > 2:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

        self.img = img_in
        self.px_size = px_size
        self.filename = filename

        self.NPS_x = None
        self.NPS_x_err = None
        self.NPS_x_f = None
        self.NPS_y = None
        self.NPS_y_err = None
        self.NPS_y_f = None

        self.S_out = np.mean(img_in.ravel())

        self.NPS_x_interp = None  # interpolated onto the MTF grid
        self.NPS_y_interp = None  # interpolated onto the MTF grid

        self.__calc_NPS() #Calculate the NPS attributes


    def __calc_NPS(self):
        """ ADD DESCRIPTION """
        width, height = self.img.shape
        img = self.img

        # ADD COMMENT
        if height * self.px_size[0] > 50.0:
            img = img[0:int(50.0 / self.px_size[0]) - 1, :]
        if width * self.px_size[1] > 50.0:
            img = img[:, 0:int(50.0 / self.px_size[1]) - 1]

        #--------------------------------------------#
        # Calculating NPS_x based on row information #
        #--------------------------------------------#
        NPS_x = 0.0 * np.abs(np.fft.fft(img[0, :]))
        print(len(NPS_x))
        print(img.shape)
        x_len = img.shape[0]
        NPS_x_arr = np.zeros((x_len, x_len))

        # ADD COMMMENT
        for j_row in np.arange(x_len):
            row = img[j_row, :]
            NPS_x_j = np.abs(2.0 * np.fft.fft(row) / len(row)) ** 2
            NPS_x_arr[j_row, :] = NPS_x_j
            NPS_x = NPS_x + NPS_x_j

        NPS_x_arr = NPS_x_arr[:, 0:int(x_len / 2)]

        self.NPS_x = np.mean(NPS_x_arr, axis=0)
        self.NPS_x_f = np.linspace(0, 0.5, len(self.NPS_x)) / self.px_size[1]
        self.NPS_x_err = np.std(NPS_x_arr, axis=0) / np.sqrt(len(self.NPS_x))

        #-----------------------------------------------#
        # Calculating NPS_y based on column information #
        #-----------------------------------------------#

        y_len = img.shape[1]
        print('y_len', y_len)
        NPS_y_arr = np.zeros((y_len, y_len))

        for j_col in np.arange(y_len):
            col = img[:, j_col]
            NPS_y_arr[j_col] = np.abs(2.0 * np.fft.fft(col) / len(col)) ** 2

        NPS_y_arr = NPS_y_arr[:, 0:int(y_len / 2)]

        self.NPS_y = np.mean(NPS_y_arr, axis=0)
        self.NPS_y_f = np.linspace(0, 0.5, len(self.NPS_y)) / self.px_size[0]
        self.NPS_y_err = np.std(NPS_y_arr, axis=0) / np.sqrt(len(self.NPS_y))



class N4255_noise_image(Test_Object_Noise):
    """ This class represents a noise image that is specifically for testing against the ANSI N42.55 standard. """

    def __init__(self, img_in, px_size, filename=None, name='noise'):
        super(N4255_noise_image, self).__init__(img_in, px_size, filename, name)

        #Attributes for flatness of field test
        self.flatness = None
        self.flatness_ROI = None
        self._cov_frac = None

        self.__calc_ff() #Calculate the Flatness of Field attributes


    def __calc_ff(self):
        """ Calculate the flatness of field for the noise image. """

        #--------------------------------#
        # Flatness of field calculations #
        #--------------------------------#
        img_flatness = self.img

        ROI_size = 170.0 / np.array(self.px_size)

        # Make sure the ROI stays within the dimensions of the image
        if ROI_size[0] >= img_flatness.shape[0]:
            ROI_size[0] = img_flatness.shape[0] - 2
        if ROI_size[1] >= img_flatness.shape[1]:
            ROI_size[1] = img_flatness.shape[1] - 2

        #---------------------------------------------------------------------------#
        # Find the center of the image and create a new ROI object to represent the #
        # flatness ROI                                                              #
        #---------------------------------------------------------------------------#
        center = np.array(img_flatness.shape) / 2
        self.flatness_ROI = ROI(img_flatness, center, ROI_size, name='flatness ROI')
        self.flatness = 1.0 - self.flatness_ROI.get_stdev() / self.flatness_ROI.get_ave()

        self._cov_frac = self.flatness_ROI.shape * np.array(self.px_size) / 170.0



class F792_noise_image(Test_Object_Noise):
    """ This class represents a noise image that is specifically for testing against the ASTM F792 standard."""

    def __init__(self, img_in, px_size, filename=None, name='noise'):
        super(F792_noise_image, self).__init__(img_in, px_size, filename, name)

        #Create any needed attributes here