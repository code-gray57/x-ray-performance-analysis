
"""
This module contains classes (structures) for holding data from testing x-ray systems against
either the ANSI N42.55 or ASTM F792 test objects.
The classes also contain methods for overall calculations of these results.
Test_Object_Data is the abstract superclass containing all the similar data between the two standards,
with a class for each standard (N4255_data and F792_data).
"""

from abc import ABC, abstractmethod

import numpy as np

from utility.MTF import MTF
from processing.calculate_img_extent import calculate_img_extent
from data_objects.test_image import N4255_image



class Test_Object_Data(ABC):
    """ This class is used to hold the data about x-ray images used to test x-ray systems
    according to different standards - specifically ANSI N42.55 and ASTM F792.

    It contains attributes for storing data for the x-ray system, the images to be tested, and for
    the tests that are similar between the two standards:
    the Organic BSNR test;
    the Steel penetration tests (though the tests are different between them);
    the Dynamic Range test;
    and the NEQ Noise test.
    """

    def __init__(self):
        """ Set the initial values (None or empty) for all the attributes used. """

        self.files = []
        self.px_size = None

        #Information about the x-ray system that is being tested
        self.xray_detector = ''
        self.xray_source = ''
        self.xray_comments = ''
        self.short_name = ''

        #The attributes for plotting/adding the results of analysis to a pdf
        self.pdf_plot = True
        self.pdf_obj = None

        #Attributes for MTF data
        self.MTF_data = MTF()

        #Attributes for Organic BSNR tests
        self.POM_boundary_signal_arr = []
        self.POM_BSNR = []

        #Attributes for steel penetration tests
        self.steel_pen_boundary_signal_arr = []
        self.steel_pen_BSNR = np.zeros(12)
        self.steel_pen = None

        #Attributes for the dynamic range tests
        self.dynamic_range = None
        self.dynamic_range_err = None

        #Attributes for the Noise (NEQ) test
        self.NEQ_x_1 = None
        self.NEQ_y_1 = None
        self.NEQ_x_1_err = None
        self.NEQ_y_1_err = None

        self.NPS_x_1 = None
        self.NPS_y_1 = None
        self.MTF_x_1 = None
        self.MTF_y_1 = None
        self.S_out = None

        self.NPS_x_interp = None
        self.NPS_y_interp = None

        #The image lists
        self.img_data = []  # a list of Std_Test_Image objects
        self.noise_img_data = []  # a list of Std_Noise_Image objects


    def add_test_img(self, test_image):
        """ Add a new test image to the image list and assign it a number. """
        test_image.img_no = len(self.img_data)
        self.img_data.append(test_image)

    def add_noise_img(self, test_noise_image):
        """ Add a new noise image to the noise image list and assign it a number. """
        test_noise_image.img_no = len(self.noise_img_data)
        self.noise_img_data.append(test_noise_image)

    def num(self, s):
        """ Convert the passed parameter into an integer.
        If it fails, convert the parameter to a float. """
        try:
            return int(s)
        except ValueError:
            return float(s)


    def read_exp_details(self, file):
        """ Reads data from a file and creates / sets attributes of the object from data in the file. """

        #Open the file
        with open(file) as f:
            content = f.read().splitlines()

        # Go through each line in the file
        for line in content:

            #The format will be:  "<attribute>=<value>"
            split = line.split('=')

            # px_size and image_area hold a pair of values, so extract each value and create a tuple from them
            if split[0] == 'px_size' or split[0] == 'image_area':
                #Remove the parenthesis that surround the two values in the file text
                str_arr = split[1].replace('(', '')
                str_arr = str_arr.replace(')', '')

                str_arr = str_arr.split(',')                       #Split at the comma to retrieve the two values
                val = (self.num(str_arr[0]), self.num(str_arr[1])) #Create a tuple with the two values (ints or floats)

                #Set the attribute (px_size or image_area) to the new tuple
                setattr(self, split[0], val)

            # Otherwise, simply set the attribute (if it exists in the object) to the value from the file
            else:
                if hasattr(self, split[0]):
                    setattr(self, split[0], split[1])


    def setup_image_data(self, plot=False, plot_type=0, system_rotated=False, aspect_corr=1.0):
        """ Set up the initial data for the images before analysis.

        :param plot: Flag for plotting the different test results. Used by the Test_Object_Image subclasses
        :param plot_type: Integer indicating what type of plot it is. Used by the Test_Object_Image subclasses
        :param system_rotated: If the system was rotated and requires correction
        :param aspect_corr: The value to correct the aspect by, if needed.
        :return: the dynamic range array, as it is different from the object's array and is required
        elsewhere in the code.
        """

        n_images = len(self.img_data)  # Number of total images it has
        dyn_range_arr = np.zeros(n_images) #Used to hold the dynamic range values for all the images

        #-----------------------------------------------------------------------------------------#
        # Loop through each image the data object has and add different test data from each image #
        # (such as the image's boundary signal, aspect ratio, etc) to an overall list.            #
        #-----------------------------------------------------------------------------------------#
        for i in range(n_images):
            # Call the analyze_image() function for the image to runs the individual calculations for the tests

            # Check if it is an N4255_image or F792_image instance - as N4255 takes an extra parameter
            # 'aspect_corr' for the Aspect Ratio test
            if isinstance(self.img_data[i], N4255_image):
                self.img_data[i].analyze_image(plot=plot, pdf_obj=self.pdf_obj, plot_type=plot_type,
                                               system_rotated=system_rotated, aspect_corr=aspect_corr)
            else:
                self.img_data[i].analyze_image(plot=plot, pdf_obj=self.pdf_obj,
                                               plot_type=plot_type, system_rotated=system_rotated)

            #----------------------------------------------------------------------------------------------#
            # Make sure to initialize the MTF data if it hasn't been already                               #
            # NOTE: This is within the for loop because it must come after an image is analyzed, otherwise #
            # the MTF_x value for the image data is None.                                                  #
            #----------------------------------------------------------------------------------------------#

            if self.MTF_data.MTF_x is None:
                self.MTF_data.MTF_x = 0.0 * self.img_data[0].MTF_obj.MTF_x
                self.MTF_data.MTF_y = 0.0 * self.img_data[0].MTF_obj.MTF_y
                self.MTF_data.MTF_x_f = self.img_data[0].MTF_obj.MTF_x_f
                self.MTF_data.MTF_y_f = self.img_data[0].MTF_obj.MTF_y_f

            # Add the MTF20 data to a list
            self.MTF_data.MTF20_x_arr.append(self.img_data[i].MTF_obj.MTF20_x)
            self.MTF_data.MTF20_y_arr.append(self.img_data[i].MTF_obj.MTF20_y)

            # Get the minimum MTF data and
            n = min(len(self.MTF_data.MTF_x),
                    len(self.MTF_data.MTF_y),
                    len(self.img_data[i].MTF_obj.MTF_x),
                    len(self.img_data[i].MTF_obj.MTF_y))

            self.MTF_data.MTF_x[0:n] += self.img_data[i].MTF_obj.MTF_x[0:n] / n_images
            self.MTF_data.MTF_y[0:n] += self.img_data[i].MTF_obj.MTF_y[0:n] / n_images

            # Add the boundary signal and steel penetration boundary signal of the image to their respective lists
            self.POM_boundary_signal_arr.append(self.img_data[i].POM_boundary_signal)
            self.steel_pen_boundary_signal_arr.append(self.img_data[i].steel_pen_BS_arr)

            # Add the dynamic range and aspect ratio of the image to their respective lists
            dyn_range_arr[i] = self.img_data[i].dynamic_range

        return dyn_range_arr #Return the dynamic range array


    def calc_dyn_range(self, dyn_range_arr):
        """ Calculate the average dynamic range and dynamic range error. """

        self.dynamic_range = np.nanmean(dyn_range_arr)
        self.dynamic_range_err = np.nanstd(dyn_range_arr)


    def calc_NEQ(self):
        """ Calculate the NEQ data.
         ADD EXTRA DETAILS HERE?
         """

        # Create zeroed out copies of MTF data for initial setup
        self.NPS_x_interp = self.MTF_data.MTF_x.copy() * 0.0
        self.NPS_y_interp = self.MTF_data.MTF_y.copy() * 0.0
        self.NPS_x_err = self.MTF_data.MTF_x.copy() * 0.0
        self.NPS_y_err = self.MTF_data.MTF_y.copy() * 0.0
        self.NEQ_x = self.MTF_data.MTF_x.copy() * 0.0
        self.NEQ_y = self.MTF_data.MTF_y.copy() * 0.0

        # Setup initial variables for the NEQ calculations
        n_noise_imgs = len(self.noise_img_data)
        NEQ_x_1_arr = []
        NEQ_y_1_arr = []
        S_out_arr = []

        #
        # Go through each noise image and
        #
        for i in range(n_noise_imgs):

            #------------------#
            # ADD COMMENT HERE #
            #------------------#

            # ADD COMMENT
            f = self.noise_img_data[i].NPS_x_f
            y = self.noise_img_data[i].NPS_x
            y_err = self.noise_img_data[i].NPS_x_err

            self.noise_img_data[i].NPS_x_interp = np.interp(self.MTF_data.MTF_x_f, f[1:], y[1:])
            self.noise_img_data[i].NPS_x_err = np.interp(self.MTF_data.MTF_x_f, f[1:], y_err[1:])

            # ADD COMMENT HERE
            self.NPS_x_interp += self.noise_img_data[i].NPS_x_interp / n_noise_imgs
            self.NPS_x_err += self.noise_img_data[i].NPS_x_err / n_noise_imgs

            # ADD COMMENT HERE
            NEQ_x_i = (self.noise_img_data[i].S_out * 1.0) ** 2 * (self.MTF_data.MTF_x ** (2)) / \
                      self.noise_img_data[i].NPS_x_interp

            self.NEQ_x = self.NEQ_x + NEQ_x_i / n_noise_imgs

            NEQ_x_1_arr.append(np.interp(1.0, self.MTF_data.MTF_x_f, NEQ_x_i))


            #------------------#
            # ADD COMMENT HERE #
            #------------------#

            # ADD COMMENT HERE
            f = self.noise_img_data[i].NPS_y_f
            y = self.noise_img_data[i].NPS_y
            y_err = self.noise_img_data[i].NPS_y_err

            self.noise_img_data[i].NPS_y_interp = np.interp(self.MTF_data.MTF_y_f, f[1:], y[1:])
            self.noise_img_data[i].NPS_y_err = np.interp(self.MTF_data.MTF_y_f, f[1:], y_err[1:])

            # ADD COMMENT HERE
            self.NPS_y_interp += self.noise_img_data[i].NPS_y_interp / n_noise_imgs
            self.NPS_y_err += self.noise_img_data[i].NPS_y_err / n_noise_imgs

            # ADD COMMENT HERE
            NEQ_y_i = (self.noise_img_data[i].S_out * 1.0) ** 2 * (self.MTF_data.MTF_y ** 2) / \
                      self.noise_img_data[i].NPS_y_interp

            self.NEQ_y = self.NEQ_y + NEQ_y_i / n_noise_imgs

            NEQ_y_1_arr.append(np.interp(1.0, self.MTF_data.MTF_y_f, NEQ_y_i))

            S_out_arr.append(self.noise_img_data[i].S_out)

        # ADD COMMENT HERE
        self.NPS_x_1 = np.interp(1.0, self.MTF_data.MTF_x_f, self.NPS_x_interp)
        self.NPS_y_1 = np.interp(1.0, self.MTF_data.MTF_y_f, self.NPS_y_interp)
        self.MTF_x_1 = np.interp(1.0, self.MTF_data.MTF_x_f, self.MTF_data.MTF_x)
        self.MTF_y_1 = np.interp(1.0, self.MTF_data.MTF_y_f, self.MTF_data.MTF_y)
        self.S_out = np.mean(S_out_arr)

        print()

        # ADD COMMENT HERE
        self.NEQ_x_1 = np.mean(NEQ_x_1_arr)
        self.NEQ_x_1_err = np.std(NEQ_x_1_arr)
        self.NEQ_y_1 = np.mean(NEQ_y_1_arr)
        self.NEQ_y_1_err = np.std(NEQ_y_1_arr)

        # Print the values for NEQ test for the user to see
        print('')
        print('NPS_x_1', self.NPS_x_1)
        print('MTF_x_1', self.MTF_x_1)
        print('NEQ_x_1_arr', NEQ_x_1_arr)

        print('')
        print('NPS_y_1', self.NPS_y_1)
        print('MTF_y_1', self.MTF_y_1)
        print('NEQ_y_1_arr', NEQ_y_1_arr)



    @abstractmethod
    def analyze_data_produce_results(self):
        """ Abstract method for analyzing all of the images held in the object and saving the results. """
        pass



class N4255_data(Test_Object_Data):
    """ This class is a specialized Test_Object_Data that holds data for the N42.55 standard.
    The main difference is the attributes for storing data about the following additional tests:
    the Aspect Ratio test;
    the Field of Flatness test;
    and the Image Extent test.
    """

    def __init__(self):
        super(N4255_data, self).__init__()

        #Attributes for the aspect ratio test
        self.aspect_ratio_arr = []
        self.aspect_ratio = None
        self.aspect_ratio_err = None

        #Attributes for the flatness of field test
        self.field_flatness = None
        self.field_flatness_err = None

        #Attributes for the image extent test
        self.image_extent = None
        self.extent_img_filename = None
        self.extent_img = None
        self.extent_dark_ROI = None
        self.extent_bright_ROI = None
        self.extent_dyn_range = None

        #Attribute to hold the area of the image
        self.image_area = None


    def add_extent_img(self, N4255_extent_image):
        """ Add an extent image to the data. """
        self.extent_img = N4255_extent_image


    def calc_aspect_ratio(self, correct_foil_asym):
        """ Calculate the aspect ratio via taking the  average from the ratios of the images used.
        Foil asymmetry can also be corrected if need be. """

        # If the foil is asymmetric, correct if via calculating it from the different aspect ratios
        # saved from the object's image files.
        if correct_foil_asym:
            asym = 0.5 * (self.aspect_ratio_arr[0] + self.aspect_ratio_arr[2] - self.aspect_ratio_arr[1] -
                          self.aspect_ratio_arr[3])
            self.aspect_ratio_arr += [-asym, asym, -asym, asym]
            self.img_data[0].aspect_ratio += -asym
            self.img_data[1].aspect_ratio += asym
            self.img_data[2].aspect_ratio += -asym
            self.img_data[3].aspect_ratio += asym

        # Get the overall aspect ratio via average and the error via standard deviation
        self.aspect_ratio = np.average(np.abs(self.aspect_ratio_arr))
        self.aspect_ratio_err = np.std(self.aspect_ratio_arr)



    def analyze_data_produce_results(self, plot=False, plot_type=0, system_rotated=False,
                                     aspect_corr=1.0, correct_foil_asym=False):
        """ Calculate the analysis results. Call the images' individual 'analyze' methods, as well as
        doing group analysis over all the images.

        :param plot: Flag for plotting the different test results. Used by the Test_Object_Image subclasses
        :param plot_type: Integer indicating what type of plot it is. Used by the Test_Object_Image subclasses
        :param system_rotated: If the system was rotated and requires correction
        :param aspect_corr: The value to correct the aspect by, if needed.
        :param correct_foil_asym: Flag for if the foil is asymmetric and needs correction.
        """

        print("Analyzing images...")
        n_images = len(self.img_data)           #Number of total images it has

        #---------------------------------------------------------------------------------------#
        # Setup the initial image data (run their "analyze_image" methods and such) and get the #
        # array of dynamic range values                                                         #
        #---------------------------------------------------------------------------------------#
        dyn_range_arr = self.setup_image_data(plot, plot_type, system_rotated, aspect_corr)

        # Since "setup_image_data" only does so on the 5 shared tests, also setup the aspect ratio values here
        for i in range(n_images):
            self.aspect_ratio_arr.append(self.img_data[i].aspect_ratio)

        #-----------------------------------------------------------------#
        # Now, go through and calculate the overall results for the tests #
        # First, start with Organic BSNR                                  #
        #-----------------------------------------------------------------#

        ave = np.average(self.POM_boundary_signal_arr)
        stdev = np.std(self.POM_boundary_signal_arr, ddof=1)
        self.POM_BSNR = ave / stdev

        # Calculate the overall MTF20 data using the average and standard deviation of the individual
        # images' MTF20 data.
        self.MTF_data.MTF20_x = np.average(self.MTF_data.MTF20_x_arr)
        self.MTF_data.MTF20_x_err = np.std(self.MTF_data.MTF20_x_arr, ddof=1)

        self.MTF_data.MTF20_y = np.average(self.MTF_data.MTF20_y_arr)
        self.MTF_data.MTF20_y_err = np.std(self.MTF_data.MTF20_y_arr, ddof=1)


        #------------------------------------------#
        # Calculate overall Steel Penetration data #
        #------------------------------------------#

        n_boundaries = len(self.steel_pen_BSNR)                     # Total number of boundaries
        steel_pen_BSNR_arr = np.zeros((n_images, n_boundaries))

        # Go through and insert the steel pen boundary signal data into the
        for i in range(n_images):
            steel_pen_BSNR_arr[i, :] = self.steel_pen_boundary_signal_arr[i]

        #
        # ADD COMMENT HERE
        #
        steel_pen = None
        for i in range(n_boundaries):

            # ADD COMMENT HERE
            data = steel_pen_BSNR_arr[:, i]
            ave = np.average(data)
            stdev = np.std(data, ddof=1)

            if stdev > 0:
                self.steel_pen_BSNR[i] = ave / stdev

            # ADD COMMENT HERE
            lim = 1.97
            if i == 0 and self.steel_pen_BSNR[i] > lim:
                steel_pen = 0

            if i > 0:
                if self.steel_pen_BSNR[i] > lim and self.steel_pen_BSNR[i - 1] > lim:
                    steel_pen = 3 * i

        self.steel_pen = steel_pen
        print('steel_pen_BSNR', self.steel_pen_BSNR)

        #-------------------------------------#
        # Calculate the overall dynamic range #
        #-------------------------------------#
        self.calc_dyn_range(dyn_range_arr)


        #-----------------------------#
        # Calculate Aspect Ratio data #
        #-----------------------------#
        self.calc_aspect_ratio(correct_foil_asym)

        #---------------------------------------#
        # Make sure the image area has been set #
        #---------------------------------------#
        if self.image_area[0] + self.image_area[1] == 0:
            self.image_area = self.img_data[0].img.shape

        #----------------------------#
        # Calculate Noise / NEQ data #
        #----------------------------#
        self.calc_NEQ()


        #--------------------------------------------------------#
        # Calculate the flatness of field using the noise images #
        #--------------------------------------------------------#

        # Go through each noise image and add their individual 'flatness' value to a list
        flatness_arr = []
        for i in range(len(self.noise_img_data)):
            flatness_arr.append(self.noise_img_data[i].flatness)

        # Get the mean and average of all the flatness values
        flatness_mean = np.mean(flatness_arr)
        flatness_stdev = np.std(flatness_arr)

        # Calculate the overall flatness for the images
        self.field_flatness = flatness_mean - 2 * flatness_stdev
        self.field_flatness_err = flatness_stdev

        #---------------------------------#
        # Calculate the image extent data #
        #---------------------------------#
        self.image_extent, self.extent_dark_ROI, self.extent_bright_ROI, self.extent_dyn_range = \
            calculate_img_extent(self.extent_img, self.px_size)

        # breakpoint()




class F792_data(Test_Object_Data):
    """ This class is a specialized Test_Object_Data that holds data for the N42.55 standard.
    The main difference are attributes for storing the data fabout the Steel Differentiation test.

    Note that it also has a Steel (useful) Penetration test - but it is separate from the N4255 test, and
    thusly has its own code.
    """

    def __init__(self):
        super(F792_data, self).__init__()

    # ADD "steel differentiation" ATTRIBUTES HERE


    def analyze_data_produce_results(self, plot=False, plot_type=0, system_rotated=False):
        """ Calculate the analysis results. Call the images' individual 'analyze' methods, as well as
        doing group analysis over all the images.

        :param plot: Flag for plotting the different test results. Used by the Test_Object_Image subclasses
        :param plot_type: Integer indicating what type of plot it is. Used by the Test_Object_Image subclasses
        :param system_rotated: If the system was rotated and requires correction
        """

        print("Analyzing images...")
        n_images = len(self.img_data)           #Number of total images it has

        #---------------------------------------------------------------------------------------#
        # Setup the initial image data (run their "analyze_image" methods and such) and get the #
        # array of dynamic range values                                                         #
        #---------------------------------------------------------------------------------------#

        dyn_range_arr = self.setup_image_data(plot, plot_type, system_rotated)

        # INCLUDE ANY SETUP FOR STEEL DIFFERENTIATION HERE
        # NOTE THAT USEFUL PENETRATION _IS_ SETUP IN ABOVE METHOD - BUT SPECIFICALLY N4255

        #-----------------------------------------------------------------#
        # Now, go through and calculate the overall results for the tests #
        # First, start with  Organic BSNR                                 #
        #-----------------------------------------------------------------#

        ave = np.average(self.POM_boundary_signal_arr)
        stdev = np.std(self.POM_boundary_signal_arr, ddof=1)
        self.POM_BSNR = ave / stdev

        # Calculate the overall MTF20 data using the average and standard deviation of the individual
        # images' MTF20 data.
        self.MTF_data.MTF20_x = np.average(self.MTF_data.MTF20_x_arr)
        self.MTF_data.MTF20_x_err = np.std(self.MTF_data.MTF20_x_arr, ddof=1)

        self.MTF_data.MTF20_y = np.average(self.MTF_data.MTF20_y_arr)
        self.MTF_data.MTF20_y_err = np.std(self.MTF_data.MTF20_y_arr, ddof=1)


        #------------------------------------------#
        # Calculate overall Steel Penetration data #
        #------------------------------------------#

        # ADD IN F792 PENETRATION TEST

        #-------------------------------------#
        # Calculate the overall dynamic range #
        #-------------------------------------#
        self.calc_dyn_range(dyn_range_arr)

        #----------------------------#
        # Calculate Noise / NEQ data #
        #----------------------------#
        self.calc_NEQ()













