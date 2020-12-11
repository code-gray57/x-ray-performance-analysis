"""
This module/package is for processing x-ray images taken of test objects for two particular standards -
ANSI N42.55 and ASTM F792 - to test the imaging performance of the systems that took the x-ray images.

The images are analyzed according to different tests outlined by the standards,
and the results are either displayed or plotted [to LaTEX to be turned into a PDF].

This main module - specifically the Xray_performance_analysis() function - sets up everything for analysis
and then calls the required functions and methods to run and complete the analysis and plot the results.
However, it does require the x-ray images used for analysis to be pre-cropped to contain only the
standard test object in the image.

Created by Jack L. Glover
"""

# Standard library imports
import glob
import pickle

# Third-party library imports
import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Package imports
from data_objects.test_data import N4255_data, F792_data
from data_objects.test_image import N4255_image, F792_image
from data_objects.test_noise import N4255_noise_image, F792_noise_image
from utility.plot_utilities import make_N4255_plots, make_F792_plots



def Xray_performance_analysis(path, is_N4255=True, pdf_plot=True, debug=False, system_rotated=False,
                              aspect_corr=1.0, log_scale=False, exp_details_file=None):
    """ Set up and process the x-ray images for testing against either the ANSI N42.55 or ASTM F792 standards.
Load in the x-ray images, prep them, and set up the required classes and variables for analysis before calling
the other classes and functions within the package to complete the actual testing.

    :param path: the path to the folder containing the images and data text files
    :param is_N4255: if the standard being tested is ANSI N42.55 (True) or ASTM F792 (False)
    :param pdf_plot: flag to plot the resulting data from the tests to a pdf file or not
    :param debug: if the debug plots and printing should be displayed or not
    :param system_rotated: If the system was rotated and requires correction
    :param aspect_corr: The value to correct the aspect by, if needed.
    :param log_scale: flag for if the images have undergone a log scale and need to reverse it
    :param exp_details_file: the path to the additional details about the x-ray system and images
    """

    # Make sure the text describing what standard is being used changes depending upon which one is wanted.
    if is_N4255:
        std_text = "ANSI N42.55"
        std_underscore = "ANSI_N4255"
    else:
        std_text = "ASTM F792"
        std_underscore = "ASTM_F792"

    print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print('  Running ' + std_text + ' analysis code')
    print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

    # Create data object - use the appropriate data object for the standard being used.
    if is_N4255:
        std_data = N4255_data()
    else:
        std_data = F792_data()

    std_data.pdf_plot = pdf_plot

    # Get list of files to analyse
    file_list_main = glob.glob1(path, "*_[1234].*")
    file_list_edge = glob.glob1(path, "*_5.*")
    file_list_blank = glob.glob1(path, "*blank.*")

    if len(file_list_main) < 4:
        file_list_main = glob.glob1(path, "*[1234].*")
        file_list_edge = glob.glob1(path, "*5.*")
        file_list_blank = glob.glob1(path, "*[67890].*")

    # file_list = file_list[0:2]
    std_data.files = [path + file for file in file_list_main]

    if exp_details_file is None:
        exp_details_file = path + glob.glob1(path, "exp_details*.txt")[0]

    std_data.read_exp_details(exp_details_file)  # Read in the details about the x-ray system

    # Assume that the folder contains images of the test object
    # in the various orientations
    pdf_filename_detailed = path + std_data.short_name + '_' + std_underscore + '_detailed_results.pdf'

    if debug and std_data.pdf_plot:
        std_data.pdf_obj = PdfPages(pdf_filename_detailed)

    #-------------------------------------------#
    # Process the main [4] images for the tests #
    #-------------------------------------------#
    print('Main test object images:')

    for file in std_data.files:
        print('  ' + file)
        img = cv2.imread(file, -1)  # Read in the file as an image

        # Scale if need be
        if log_scale:
            img = undo_log_scale(img)

        #-------------------------------------------------------------------------------------#
        # Set the created image as an image object, and set up some initial attributes for it #
        #-------------------------------------------------------------------------------------#
        if is_N4255:
            std_img = N4255_image(img)
        else:
            std_img = F792_image(img)

        std_img.px_size = std_data.px_size
        std_img.filename = file
        std_img.img_no = int(file.split('.')[-2][-1:])
        std_img.orientation = int(file.split('.')[-2][-1:])  # last no before .tif = orientation

        std_img.set_ROIs(pdf=std_data.pdf_obj, debug=debug)  # determine ROIs
        std_data.add_test_img(std_img)  # add image to test data obj

    #----------------------------------#
    # Process the blank [noise] images #
    #----------------------------------#
    print('Blank images:')
    for blank_file in file_list_blank:

        print('  ' + path + blank_file)
        img = cv2.imread(path + blank_file, -1)  # Load in the blank image from the file

        # Scale down if need be.
        if log_scale:
            img = undo_log_scale(img)

        #--------------------------------------------------------------------------#
        # Set the image as a noise image object and set up some initial attributes #
        #--------------------------------------------------------------------------#
        if is_N4255:
            noise_img = N4255_noise_image(img, std_data.px_size, filename=path + blank_file,
                                          name=std_data.short_name)
        else:
            noise_img = F792_noise_image(img, std_data.px_size, filename=path + blank_file,
                                         name=std_data.short_name)

        std_data.add_noise_img(noise_img)

    #-------------------------------------------------------------------------------------------#
    # If ANSI N42.55 is the standard to test against, load in the extent image for the 7th test #
    #-------------------------------------------------------------------------------------------#
    if is_N4255:
        std_data.extent_img_filename = path + file_list_edge[0]
        print('Test 7 image:')
        print('  ' + std_data.extent_img_filename)
        extent_img = cv2.imread(std_data.extent_img_filename, -1)

        # Scale if need be
        if log_scale:
            extent_img = undo_log_scale(extent_img)

        std_data.add_extent_img(extent_img)  # Add the image to the data object

    #========================#
    # Determine the results! #
    #========================#

    # If the standard is N42.55, then an additional parameter - aspect_corr - should be passed
    # into the 'analyze_data_produce_results' method.
    if is_N4255:
        std_data.analyze_data_produce_results(plot=debug, system_rotated=system_rotated, aspect_corr=aspect_corr)
    else:
        std_data.analyze_data_produce_results(plot=debug, system_rotated=system_rotated)

    if debug and std_data.pdf_plot:
        std_data.pdf_obj.close()

    # Write the data to a CSV file
    csv_filename = std_data.short_name + '_' + std_underscore + '_summary.p'
    write_data_to_csv(std_data, csv_filename)

    #-----------------------------------------------------------------------------------#
    # Save the results gathered and plotted within the pdf object as an actual pdf file #
    #-----------------------------------------------------------------------------------#
    pdf_filename_summary = path + std_data.short_name + '_' + std_underscore + '_summary.pdf'

    # If the pdf creation is specified, set up the object for it
    if std_data.pdf_plot:
        std_data.pdf_obj = PdfPages(pdf_filename_summary)

    # Plot the test results and accompanying images, graphs, etc. to the pdf object
    if is_N4255:
        make_N4255_plots(std_data, aspect_corr=aspect_corr)
    else:
        make_F792_plots(std_data)

    if std_data.pdf_plot:  # Close the pdf object if it was opened/created
        std_data.pdf_obj.close()

    print('Analysis complete')



def undo_log_scale(img):
    """ Undoes a log scale on the passed image. """
    zero = 0
    img = np.exp(5.0 * (img.astype(np.float64) - zero) / 64000.0)
    img_max = np.max(img.ravel())
    img = img * 60000.0 / img_max
    img = img.astype(np.int32)

    return img


def write_data_to_csv(std_data, out_filename):
    """ Saves the data from the tests to a .csv file.

    :param std_data: the object (Test_Object_Data or subclass) that holds the test results data.
    :param out_filename: the name of the .csv file to write the data to
    """

    # All the attributes to be saved from the data object
    save_attributes = ['px_size', 'MTF20_x', 'MTF20_y', 'MTF_x', 'MTF_y', 'MTF_x_f', 'MTF_y_f',
                       'steel_pen', 'dynamic_range', 'POM_BSNR',
                       'NPS_x_interp', 'NPS_y_interp', 'NEQ_x_1', 'NEQ_y_1', 'NEQ_x', 'NEQ_y']

    save_dict = {'filename': out_filename}  # The dictionary to store the wanted attributes before saving to the file

    # For each attribute name in the list, get the attribute from the data object and save it to the dictionary
    for attr in save_attributes:

        # Added code to fix the problem when MTF data is put into one class instantiation
        if attr.startswith("MTF"):
            save_dict[attr] = getattr(std_data.MTF_data, attr)
        else:
            save_dict[attr] = getattr(std_data, attr)

    # Dump the dictionary into the actual .csv file
    with open(out_filename, 'wb') as output:
        pickle.dump(save_dict, output, pickle.HIGHEST_PROTOCOL)



#===========================================================================================#
# If the program is run as the main executable, then default to using the example folder of #
# images and data to do a test run of the processing.                                       #
#===========================================================================================#
if __name__ == '__main__':
    Xray_performance_analysis('example_images/')




