"""
This module contains classes that represent x-ray images to be used for analysis. Each class contains
attributes to hold data for both the image it loads in (its file name, orientation, etc) as well as
some data generated from and/or for analysis (such as boundary signals and ROIs).

It contains 3 classes, one superclass that contains the majority of the methods and attributes, with
two subclasses that cater specifically to either the ANSI N42.55 or ASTM F792 standard.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utility.ROI import ROI
from processing.calc_boundary_sig import calc_boundary_sig
from utility.MTF import MTF, determine_MTF_20, MTF_from_slanted_sq, get_MTF_data
from processing.calculate_dyn_range import calculate_dyn_range
from processing.calc_steel_pen_N4255 import calc_steel_pen_N4255


class Test_Object_Image(ABC):
    """
    This abstract class is a container to hold an x-ray image for use in
    testing an x-ray system against a standard.

    It contains data about the image (file name, pixel size in mm, etc) as well as
    data concerning the actual tests (such as the dynamic range for the image).
    """

    def __init__(self, img):

        #Set attributes for general data about the image.
        self.filename = ''
        self.img_no = None
        self.px_size = (0.0, 0.0)  # in mm
        self.orientation = None

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.img = img

        # For Spatial Resolution test, which is shared between N4255 and F792
        self.MTF_obj = MTF()

        self.lead_foil_ROI = None # Also for Aspect Ratio?

        # For Organic BSNR
        self.POM_piece_ROI = None
        self.POM_boundary_signal = None

        # For Dynamic Range
        self.dynamic_range = None
        self.dynamic_range_dark_ROI = None
        self.dynamic_range_light_ROI = None

        # For Steel Penetration (and Dynamic Range)
        self.steel_pen_BS_arr = []
        self.step_wedge_boundaries = None
        self.step_wedge_ROI = None



    def find_foil_center_by_blur(self):
        """ Calculate the center of the foil test piece by blurring the image. """
        r_blur = int(self.img.shape[0] / 6)

        blurred = self.img * 1.0

        max = np.max(blurred) #Get the maximum value of the image

        # Max out the edges of the image
        blurred[0: int(self.img.shape[0] * 0.25), :] = max
        blurred[int(self.img.shape[0] * 0.6):, :] = max
        blurred[:, int(self.img.shape[1] * 0.45):] = max
        blurred[:, 0:int(self.img.shape[1] * 0.1)] = max

        #Now call the actual function to blur the image [and get the index of the minimum value in the image]
        blurred = cv2.blur(blurred, (r_blur, r_blur))
        foil_center_pos = np.unravel_index(blurred.argmin(), blurred.shape)

        return foil_center_pos


    def set_ROIs(self, pdf=None, debug=False):
        """ Set up the three main ROIs of the image - the foil, the POM piece, and the step wedge ROIs.

        :param pdf: a pdf object to save the ROIs to, if specified. Otherwise display them to the user.
        :param debug: flag to display the ROIs regardless of the 'pdf' parameter.
        """

        # Check if the image needs to be horizontally flipped
        left_ave = np.mean(self.img[int(self.img.shape[0] * 0.4):int(self.img.shape[0] * 0.6),
                           int(self.img.shape[1] * 0.2):int(self.img.shape[1] * 0.3)].ravel())
        right_ave = np.mean(self.img[int(self.img.shape[0] * 0.4):int(self.img.shape[0] * 0.6),
                            int(self.img.shape[1] * 0.7):int(self.img.shape[1] * 0.8)].ravel())

        # See which average is greater and flip the image respectively
        if left_ave < right_ave:
            print('  horizontally flipped')
            self.img = fimg = cv2.flip(self.img, 1)

        px_size0 = self.px_size[0]
        px_size1 = self.px_size[1]


        #------------------------------------------------------------#
        # Calculate the ROIs for the step wedge, foil, and POM piece #
        #------------------------------------------------------------#

        # ADD COMMENT HERE
        center = (self.img.shape[0] * 0.5, self.img.shape[1] * 0.75)
        size = (15 * 15 / px_size1, 25 / px_size0)
        self.step_wedge_ROI = ROI(self.img, center, size, name='full step wedge ROI', store_img=True)

        # ADD COMMENT HERE
        center = self.find_foil_center_by_blur()
        size = (55 / px_size1, 55 / px_size0)
        self.lead_foil_ROI = ROI(self.img, center, size, name='full foil ROI', store_img=True)

        # ADD COMMENT HERE
        center = (self.lead_foil_ROI.center[0] + 1.05 * 76.0 / px_size0,
                  self.lead_foil_ROI.center[1] - 1.05 * 6.0 / px_size1)
        size = (30 / px_size1, 15 / px_size0)
        self.POM_piece_ROI = ROI(self.img, center, size, name='POM piece ROI', store_img=True)

        #-----------------------------------------------------------#
        # Either save the ROIs to a pdf or display them as a figure #
        #-----------------------------------------------------------#
        if pdf is not None or debug:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
            plt.imshow(self.img, cmap='Greys_r')

            self.step_wedge_ROI.add_rect_to_plot()
            self.lead_foil_ROI.add_rect_to_plot()
            self.POM_piece_ROI.add_rect_to_plot()

            # Display and/or save the ROI plots to a pdf
            if debug:
                plt.show()
                fig = plt.figure(figsize=(8.27, 11.69), dpi=100)

            if pdf is None:
                plt.show()
            else:
                pdf.savefig()

            plt.close(fig)

    @abstractmethod
    def analyze_image(self):
        """ Abstract method for running analysis on the image stored within the class instance. """
        pass



class N4255_image(Test_Object_Image):
    """ This class represents an x-ray image being used to test an x-ray system against the N42.55 standard.
     The main difference from the superclass is that it contains an attribute for the Aspect Ratio test. """

    def __init__(self, img):
        super(N4255_image, self).__init__(img) #Call the superclass' __init__ method

        # For Aspect Ratio
        self.aspect_ratio = None


    #Implement the abstract method inherited from its superclass
    def analyze_image(self, plot=False, pdf_obj=None, plot_type=0, system_rotated=False, aspect_corr=1.0):
        """ Run individual analysis functions on the image data and
        update the attributes to reflect the results.

        :param plot: Flag for plotting data - whether directly to the user or to a pdf
        :param pdf_obj: The object representing a pdf object to plot to if plot is True
        :param plot_type:
        :param system_rotated: If the system was rotated and requires correction
        :param aspect_corr: The value to correct the aspect ratio by if needed.
        """

        #--------------#
        # Organic BSNR #
        #--------------#

        ROI_size_BSNR = 2.0
        filename = 'img' + str(self.img_no) + 'BSNR.png'

        # Calculate the boundary signal
        self.POM_boundary_signal = calc_boundary_sig(self.POM_piece_ROI, self.px_size,
                                                                 ROI_size_BSNR, plot=plot)

        #--------------------#
        # Spatial Resolution #
        #--------------------#

        ROI_size_MTF = 20.0
        ROI_size_px = (ROI_size_MTF / self.px_size[0], ROI_size_MTF / self.px_size[1])
        MTF_data, freq_data = MTF_from_slanted_sq(self.lead_foil_ROI.get_img(),
                                                                self.lead_foil_ROI.size)

        # Check the orientation and flip the x and y '_ind' values if need be to match [so that
        # the orientations all match up despite coming in at different rotations]
        if self.orientation == 1 or self.orientation == 3 or system_rotated:
            x_ind = 0
            y_ind = 1
        elif self.orientation == 2 or self.orientation == 4:
            x_ind = 1
            y_ind = 0
        else:
            print("An error occured in N4255_image.analyze_image(): self.orientation and/or system_rotated invalid.")

        # Calculate the MTF data
        self.MTF_obj.MTF_x_f, self.MTF_obj.MTF_x = get_MTF_data(MTF_data[x_ind],
                                                                freq_data[x_ind] / self.px_size[1])

        self.MTF_obj.MTF20_x = determine_MTF_20(self.MTF_obj.MTF_x,
                                                self.MTF_obj.MTF_x_f,
                                                plot_type=plot_type)

        self.MTF_obj.MTF_y_f, self.MTF_obj.MTF_y = get_MTF_data(MTF_data[y_ind],
                                                                freq_data[y_ind] / self.px_size[0])

        self.MTF_obj.MTF20_y = determine_MTF_20(self.MTF_obj.MTF_y,
                                                self.MTF_obj.MTF_y_f,
                                                plot_type=plot_type)

        #--------------------#
        # Useful penetration #
        #--------------------#
        ROI_size_mm = 2.0
        ROI_size_px = [ROI_size_mm / self.px_size[0], ROI_size_mm / self.px_size[1]]

        self.step_wedge_boundaries, self.steel_pen_BS_arr = calc_steel_pen_N4255(
                                                                self.step_wedge_ROI.img,
                                                                ROI_size_px,
                                                                px_size=self.px_size,
                                                                plot=plot,
                                                                pdf_obj=pdf_obj)

        #---------------#
        # Dynamic range #
        #---------------#

        ROI_size_mm = 1.0
        ROI_size_px = [ROI_size_mm / self.px_size[0], ROI_size_mm / self.px_size[1]]

        self.dynamic_range, self.dynamic_range_dark_ROI, self.dynamic_range_light_ROI = calculate_dyn_range(
                                                                                            self.img,
                                                                                            self.step_wedge_ROI,
                                                                                            self.step_wedge_boundaries,
                                                                                            ROI_size_px,
                                                                                            export_ROIs=True)

        #--------------#
        # Aspect Ratio #
        #--------------#
        self.aspect_ratio = self.measure_aspect_ratio_from_foil(aspect_corr=aspect_corr)



    #Should remain in N4255 as Aspect Ratio is not a part of the F792 tests
    def measure_aspect_ratio_from_foil(self, plot=False, subplot=(4, 3, 1), aspect_corr=1.0):
        """ ADD GENERAL DESCRIPTION OF METHOD

        :param plot: flag for creating and displaying plots immediately for some of the values
        :param subplot: tuple for the subplots when plotting (if plot is True)
        :param aspect_corr: the value to correct the aspect ratio if need be
        """

        shape = self.lead_foil_ROI.get_img().shape

        #-------------------------#
        # Horizontal aspect ratio #
        #-------------------------#

        # ADD COMMENT
        horiz_lineout = self.lead_foil_ROI.get_img()[int(shape[0] / 2), :]
        height_change = np.mean(horiz_lineout[-11:-1]) - np.mean(horiz_lineout[0:10])
        horiz_lineout = horiz_lineout - height_change * np.linspace(0.0, 1.0, num=len(horiz_lineout))

        # ADD COMMENT
        hi_val = 0.5 * (np.mean(horiz_lineout[3:10]) + np.mean(horiz_lineout[-10:-3]))
        lo_val = np.mean(horiz_lineout[int(len(horiz_lineout) * 0.5 - 5):int(len(horiz_lineout) * 0.5 + 5)])
        mid_val = 0.5 * (lo_val + hi_val)

        # ADD COMMENT
        n = len(horiz_lineout)
        ind1 = np.interp(mid_val, 2 * mid_val - horiz_lineout[0:int(len(horiz_lineout) / 2)],
                         np.arange(int(len(horiz_lineout) / 2)))

        other_end = horiz_lineout[int(len(horiz_lineout) / 2):]
        ind2 = np.interp(mid_val, other_end, np.arange(len(other_end))) + len(horiz_lineout) / 2

        d_horiz = ind2 - ind1

        # ADD COMMENT ABOUT PLOTTING
        if plot:
            plt.subplot(subplot[0], subplot[1], subplot[2])
            plt.xticks([])
            plt.yticks([])
            plt.title('pixel values along blue line', fontsize=8, color='b')
            plt.plot(horiz_lineout, color='black')
            plt.plot((ind1), (mid_val), 'o', color='black')
            plt.plot((ind2), (mid_val), 'o', color='black')

        #-----------------------#
        # Vertical aspect ratio #
        #-----------------------#

        # ADD COMMENT
        vert_lineout = self.lead_foil_ROI.get_img()[:, int(shape[1] / 2)]
        n = len(vert_lineout)
        height_change = np.mean(vert_lineout[-11:-1]) - np.mean(vert_lineout[0:10])
        vert_lineout = vert_lineout - height_change * np.linspace(0.0, 1.0, num=len(vert_lineout))

        # ADD COMMENT
        hi_val = 0.5 * (np.mean(vert_lineout[3:10]) + np.mean(vert_lineout[-10:-3]))
        lo_val = np.mean(vert_lineout[int(len(vert_lineout) / 2 - 5):int(len(vert_lineout) / 2 + 5)])
        mid_val = 0.5 * (lo_val + hi_val)

        # ADD COMMENT
        len1 = int(len(vert_lineout) / 2)
        ind1 = np.interp(mid_val, 2 * mid_val - vert_lineout[0:len1], np.arange(len1))

        other_end = vert_lineout[int(len(vert_lineout) / 2):]
        ind2 = np.interp(mid_val, other_end, np.arange(len(other_end))) + len(vert_lineout) / 2

        d_vert = ind2 - ind1
        d_vert = d_vert * aspect_corr

        # ADD COMMENT ABOUT PLOTTING
        if plot:
            plt.subplot(subplot[0], subplot[1], subplot[2] + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('pixel values along red line', fontsize=8, color='r')
            plt.plot(vert_lineout, color='black')
            plt.plot((ind1), (mid_val), 'o', color='black')
            plt.plot((ind2), (mid_val), 'o', color='black')

        return np.abs(1.0 - d_vert * 1.0 / d_horiz)



class F792_image(Test_Object_Image):
    """ This class represents an x-ray image being used to test an x-ray machine against the F792 standard.
    The main difference from the superclass is that it contains attribute(s) for the Steel Differentation test.
    """

    def __init__(self, img):
        super(F792_image, self).__init__(img) #Call the superclass' __init__ method

        # For Steel Differentiation attributes


    # INCLUDE ANY NEEDED METHODS FOR STEEL DIFFERENTIATION AND USEFUL PENETRATION CALCULATIONS



    # Implement the abstract method inherited from its superclass
    def analyze_image(self, plot=False, pdf_obj=None, plot_type=0, system_rotated=False):
        """ Run individual analysis functions on the image data and
        update the attributes to reflect the results.

        :param plot: Flag for plotting data - whether directly to the user or to a pdf
        :param pdf_obj: The object representing a pdf object to plot to if plot is True
        :param plot_type:
        :param system_rotated: If the system was rotated and requires correction
        """

        #-----------------------#
        # Steel Differentiation #
        #-----------------------#

        # ADD CODE OR FUNCTION CALL HERE


        #--------------#
        # Organic BSNR #
        #--------------#

        ROI_size_BSNR = 2.0
        filename = 'img' + str(self.img_no) + 'BSNR.png'

        # Calculate the boundary signal
        self.POM_boundary_signal = calc_boundary_sig(self.POM_piece_ROI, self.px_size,
                                                                 ROI_size_BSNR, plot=plot)

        #--------------------#
        # Spatial Resolution #
        #--------------------#

        ROI_size_MTF = 20.0
        ROI_size_px = (ROI_size_MTF / self.px_size[0], ROI_size_MTF / self.px_size[1])
        MTF_data, freq_data = MTF_from_slanted_sq(self.lead_foil_ROI.get_img(), self.lead_foil_ROI.size)

        # Check the orientation to make sure
        if self.orientation == 1 or self.orientation == 3 or system_rotated:
            x_ind = 0
            y_ind = 1
        elif self.orientation == 2 or self.orientation == 4:
            x_ind = 1
            y_ind = 0
        else:
            print("An error occured in N4255_image.analyze_image(): self.orientation and/or system_rotated invalid.")

        # ADD COMMENT HERE
        self.MTF_obj.MTF_x_f, self.MTF_obj.MTF_x = get_MTF_data(MTF_data[x_ind],
                                                                    freq_data[x_ind] / self.px_size[1])

        self.MTF_obj.MTF20_x = determine_MTF_20(self.MTF_obj.MTF_x,
                                                self.MTF_obj.MTF_x_f,
                                                plot_type=plot_type)

        self.MTF_obj.MTF_y_f, self.MTF_obj.MTF_y = get_MTF_data(MTF_data[y_ind],
                                                                    freq_data[y_ind] / self.px_size[0])

        self.MTF_obj.MTF20_y = determine_MTF_20(self.MTF_obj.MTF_y,
                                                self.MTF_obj.MTF_y_f,
                                                plot_type=plot_type)

        #--------------------#
        # Useful penetration #
        #--------------------#

        # Insert F792 useful penetration code (or function call if code has its own function) here
        # NOTE: the 'calc_dyn_range()' for the Dynamic Range test later in the code will cause an error
        #       unless the 'self.step_wedge_boundaries' attribute is set - which in N4255 code it is set with
        #       the call to 'calc_steel_pen_N4255' (along with self.steel_pen_BS_arr)

        # -- N4255 code for steel penetration --
        # ROI_size_mm = 2.0
        # ROI_size_px = [ROI_size_mm / self.px_size[0], ROI_size_mm / self.px_size[1]]
        #
        # self.step_wedge_boundaries, self.steel_pen_BS_arr = calc_steel_pen_N4255(self.step_wedge_ROI.img,
        #                                                                          ROI_size_px,
        #                                                                          px_size=self.px_size,
        #                                                                          plot=plot,
        #                                                                          pdf_obj=pdf_obj)

        #---------------#
        # Dynamic range #
        #---------------#

        ROI_size_mm = 1.0
        ROI_size_px = [ROI_size_mm / self.px_size[0], ROI_size_mm / self.px_size[1]]
        self.dynamic_range, self.dynamic_range_dark_ROI, self.dynamic_range_light_ROI = calculate_dyn_range(
                                                                                            self.img,
                                                                                            self.step_wedge_ROI,
                                                                                            self.step_wedge_boundaries,
                                                                                            ROI_size_px,
                                                                                            export_ROIs=True)



