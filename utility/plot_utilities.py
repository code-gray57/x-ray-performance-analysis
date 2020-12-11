
"""
This module contains functions to plot the results of x-ray system performance analysis -
specifically in accordance with ANSI N42.55 and ASTM F792 standards - to [LaTEX and then to a PDF]

The two main functions are 'make_N4255_plots' and 'make_F792_plots', while the rest are called from
these to plot individual pieces of the analysis results.
"""

#std library imports
import time

#Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

#Package imports
from utility.ROI import ROI
from processing.calc_boundary_sig import calc_boundary_sig


def make_N4255_plots(data_obj, aspect_corr=1.0, title_pages=False):
    """ This function creates a new pdf and plots all of the results from analyzing the specified images
        of an x-ray system according to the ANSI N42.55 standard.

        The nine tests that are plotted are as follows:
        Steel Penetration;
        Organic BSNR;
        Spatial Resolution;
        Dynamic Range;
        Noise (NEQ);
        Field of Flatness;
        Image Extent;
        Image Area;
        Aspect Ratio

        :param data_obj: the object (Test_Object_Data or subclass) that holds all of the test result data
        :param aspect_corr: the value to correct the aspect ratio by if need be
        :param title_pages: Flag for if a title page should be created before each of the test results
        """

    print("Generating plots...")

    #Create color maps
    cmap = plt.get_cmap('jet')
    cmap = plt.get_cmap('gray')

    #Call the function to create the title page of the pdf document
    plot_front_title(data_obj)

    #-----------------------------------------------------------------------#
    # Initialize the position variables for the text and graphs on the pdf. #
    #-----------------------------------------------------------------------#
    y0 = 0.9
    dy = [0.03, 0.025]

    ha = 'left'
    va = 'center'
    fs = 10
    dfs = 2

    #  metric name     value     unc    min
    xpos = [0.0, 0.4, 0.5, 0.75]
    yi = y0 - 0.1  # The position of the text on the y access, which is constantly updated as more text is added

    #-----------------------------------------------------------------------------------#
    # Plot the 'summary' page listing all the tests and the overall results - TEXT ONLY #
    #-----------------------------------------------------------------------------------#

    #Create the title of the page
    plot_overall_text(data_obj, yi, xpos, ha, va, fs)

    #Plot the overall results text of the first test, Penetration
    yi = yi - dy[0]
    plot_pen_text(data_obj, 1, yi, xpos, ha, va, fs, dfs)

    #Plot the overall results text of the first test, Organic Material Detection
    yi = yi - dy[0]
    plot_BSNR_text(data_obj, 2, yi, xpos, ha, va, fs, dfs)

    #Plot the overall results text of the third test, Spatial Resolution
    yi = yi - dy[0]
    plot_spatial_text(data_obj, 3, yi, yi - dy[1], xpos, ha, va, fs, dfs)
    yi = yi - dy[1] #Make sure the local yi is updated

    #Plot the overall results text of the fourth test, Dynamic Range
    yi = yi - dy[0]
    plot_dyn_text(data_obj, 4, yi, xpos, ha, va, fs, dfs)

    #Plot the overall results text of the fifth test, NEQ Noise
    yi = yi - dy[0]
    plot_noise_text(data_obj, 5, yi, dy, xpos, ha, va, fs, dfs)
    yi = yi - (dy[1] * 2) #Make sure to update yi, as it was only locally changed in 'plot_noise_text()'

    #Plot the overall results text of the sixth test, Flatness of field
    yi = yi - dy[0]
    plot_ff_text(data_obj, 6, yi, xpos, ha, va, fs, dfs)

    #Plot the overall results text of the seventh test, Image Extent
    yi = yi - dy[0]
    plot_extent_text(data_obj, 7, yi, xpos, ha, va, fs, dfs)

    #Plot the overall results text of the eighth test, Image Area
    yi = yi - dy[0]
    plot_area_text(data_obj, 8, yi, xpos, ha, va, fs, dfs)

    #Plot the overall results text of the ninth test, Aspect Ratio
    yi = yi - dy[0]
    plot_a_ratio_text(data_obj, 9, yi, xpos, ha, va, fs, dfs)

    #--------------------------------------------------#
    # Plot the footnotes for the overall results page. #
    #--------------------------------------------------#
    plot_overall_footnotes(xpos, ha, va, fs, dfs)


    #-----------------#
    # Plot the images #
    #-----------------#
    plot_images(data_obj, fs) #Plot the images to the pdf

    plot_image_footnotes(data_obj, xpos, ha, va, fs, dfs) #Add in the footnotes to the pdf


    #-------------------#
    # Penetration plots #
    #-------------------#
    if title_pages:
        new_title_page(data_obj, "Test 1: Penetration")

    #Call the function to plot the Steel Penetration results to the pdf
    plot_steel_pen_N4255(data_obj, 1)


    #------------#
    # BSNR plots #
    #------------#
    if title_pages:
        new_title_page(data_obj, "Test 2: Organic Material Detection")

    # Call the function to plot the Organic Material Detection results to the pdf
    plot_BSNR(data_obj, 2, cmap)


    #--------------------#
    # Spatial Resolution #
    #--------------------#
    if title_pages:
        new_title_page(data_obj, "Test 3: Spatial Resolution")

    # Call the function to plot the Spatial Resolution results to the pdf
    plot_spatial_res(data_obj, 3)

    #---------------#
    # Dynamic Range #
    #---------------#
    if title_pages:
        new_title_page(data_obj, "Test 4: Dynamic Range")

    # Call the function to plot the Dynamic Range results to the pdf
    plot_dynamic_range(data_obj, 4)

    #-------#
    # Noise #
    #-------#
    if title_pages:
        new_title_page(data_obj, "Test 5: Noise (NEQ)")

    # Call the function to plot the Noise (NEQ) results to the pdf
    plot_noise(data_obj, 5)

    #-------------------#
    # Flatness of field #
    #-------------------#
    if title_pages:
        new_title_page(data_obj, "Test 6: Flatness of Field")

    # Call the function to plot the Flatness of Field results to the pdf
    plot_field_flatness(data_obj, 6)

    #--------------#
    # Image extent #
    #--------------#
    if title_pages:
        new_title_page(data_obj, "Test 7: Image Extent")

    # Call the function to plot the Image Extent results to the pdf
    plot_image_extent(data_obj, 7)


    #------------#
    # Image Area #
    #------------#
    if title_pages:
        fig = new_pdf_page(data_obj.pdf_obj)
        plt.axis('off')
        plt.text(0.5, 0.5, 'Test 8: Image Area', ha='center', va='center', fontsize=20)
        str1 = str(data_obj.image_area[0]) + ' by ' + str(data_obj.image_area[1]) + ' pixels'
        plt.text(0.5, 0.4, str1, ha='center', va='center', fontsize=12)

    #--------------#
    # Aspect Ratio #
    #--------------#
    if title_pages:
        new_title_page(data_obj, "Test 9: Aspect Ratio")

    #Call the function to plot the Aspect Ratio results to the pdf
    plot_aspect_ratio(data_obj, 9, cmap, aspect_corr)

    fig = new_pdf_page(data_obj.pdf_obj, open_fig=False)



def make_F792_plots(data_obj, title_pages=False):
    """ This function creates a new pdf and plots all of the results from analyzing the specified images
    of an x-ray system according to the ASTM F792 standard.

    The six tests that are plotted are as follows:
    Steel Differentiation;
    Steel Penetration;
    Organic BSNR;
    Spatial Resolution;
    Dynamic Range;
    Noise (NEQ)

    :param data_obj: the object (Test_Object_Data or subclass) that holds all of the test result data
    :param title_pages: Flag for if a title page should be created before each of the test results
    """

    print("Generating plots...")

    # Create color maps
    cmap = plt.get_cmap('jet')
    cmap = plt.get_cmap('gray')

    # Call the
    plot_front_title(data_obj)

    # -----------------------------------------------------------------------#
    # Initialize the position variables for the text and graphs on the pdf. #
    # -----------------------------------------------------------------------#
    y0 = 0.9
    dy = [0.03, 0.025]

    ha = 'left'
    va = 'center'
    fs = 10
    dfs = 2

    #  metric name     value     unc    min
    xpos = [0.0, 0.4, 0.5, 0.75]
    yi = y0 - 0.1  # The position of the text on the y access, which is constantly updated as more text is added

    # -----------------------------------------------------------------------------------#
    # Plot the 'summary' page listing all the tests and the overall results - TEXT ONLY #
    # -----------------------------------------------------------------------------------#

    # Create the title of the page
    plot_overall_text(data_obj, yi, xpos, ha, va, fs)

    #Plot the overall results text of the first test, Steel Differentiation


    # Plot the overall results text of the second test, Penetration
    yi = yi - dy[0]
    plot_pen_text(data_obj, 2, yi, xpos, ha, va, fs, dfs)

    # Plot the overall results text of the third test, Organic Material Detection
    yi = yi - dy[0]
    plot_BSNR_text(data_obj, 3, yi, xpos, ha, va, fs, dfs)

    # Plot the overall results text of the fourth test, Spatial Resolution
    yi = yi - dy[0]
    plot_spatial_text(data_obj, 4, yi, yi - dy[1], xpos, ha, va, fs, dfs)
    yi = yi - dy[1]  # Make sure the local yi is updated

    # Plot the overall results text of the fifth test, Dynamic Range
    yi = yi - dy[0]
    plot_dyn_text(data_obj, 5, yi, xpos, ha, va, fs, dfs)

    # Plot the overall results text of the sixth test, Noise
    yi = yi - dy[0]
    plot_noise_text(data_obj, 6, yi, dy, xpos, ha, va, fs, dfs)
    yi = yi - (dy[1] * 2)  # Make sure to update yi, as it was only locally changed in 'plot_noise_text()'

    # --------------------------------------------------#
    # Plot the footnotes for the overall results page. #
    # --------------------------------------------------#
    plot_overall_footnotes(xpos, ha, va, fs, dfs, standard="ASTM F792")


    #---------------------------------------------------------#
    # Plot the cropped and rotated images from the processing #
    #---------------------------------------------------------#
    plot_images(data_obj, fs)  # Plot the images to the pdf

    plot_image_footnotes(data_obj, xpos, ha, va, fs, dfs)  # Add in the footnotes to the pdf

    # NOTE: Above image plotting the same, with the same footnotes, for F792???

    #-----------------------------#
    # Steel differentiation plots #
    #-----------------------------#
    if title_pages:
        new_title_page(data_obj, "Test 1: Steel Differentiation")

    #Call the function to plot the Steel Differentiation results to the pdf


    #-------------------#
    # Penetration plots #
    #-------------------#
    if title_pages:
        new_title_page(data_obj, "Test 2: Penetration")

    # Call the function to plot the Steel Penetration results to the pdf
    #plot_steel_pen(data_obj, 2)

    #------------#
    # BSNR plots #
    #------------#
    if title_pages:
        new_title_page(data_obj, "Test 3: Organic Material Detection")

    # Call the function to plot the Organic Material Detection results to the pdf
    plot_BSNR(data_obj, 3, cmap)

    #--------------------#
    # Spatial Resolution #
    #--------------------#
    if title_pages:
        new_title_page(data_obj, "Test 4: Spatial Resolution")

    # Call the function to plot the Spatial Resolution results to the pdf
    plot_spatial_res(data_obj, 4)

    #---------------#
    # Dynamic Range #
    #---------------#
    if title_pages:
        new_title_page(data_obj, "Test 5: Dynamic Range")

    # Call the function to plot the Dynamic Range results to the pdf
    plot_dynamic_range(data_obj, 5)

    #-------#
    # Noise #
    #-------#
    if title_pages:
        new_title_page(data_obj, "Test 6: Noise (NEQ)")

    # Call the function to plot the Noise (NEQ) results to the pdf
    plot_noise(data_obj, 6)

    fig = new_pdf_page(data_obj.pdf_obj, open_fig=False)



#=================================================#
# Some utility functions used to create pdf pages #
#=================================================#

def new_pdf_page(pdf_obj, close_fig = True, open_fig = True, pdf_plot = True):
    """ Create a new page within the passed "pdf_obj" PdfPages argument.

    :param pdf_obj: the object to create the new page to
    :param close_fig: flag to save the pdf or  display it onto the screen
    :param open_fig: flag to simply create a new figure and return it
    :param pdf_plot: Whether or not the new page should be actually plotted to a pdf or simply displayed.
    :return: None, or the figure if open_fig is set to True
    """
    if close_fig:
        if pdf_plot:
            pdf_obj.savefig()
        else:
            plt.show()
        plt.close()

    if open_fig:
        fig = plt.figure(figsize=(8.5, 11))
        return fig
    else:
        return None


def new_title_page(data_obj, text):
    """ Creates a new title page in the pdf object of the passed 'data_obj'

    :param data_obj: the data object that contains the pdf object to save the new title page to
    :param text: the text of the title page.
    """

    new_pdf_page(data_obj.pdf_obj)
    plt.axis('off')
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=20)


#=================================================================================#
# The functions that plot the overall results text of the ran tests to the pdf.   #
# These do not create any graphs or show the images of the said tests: they only  #
# create text onto the current pdf page about the specific test they are made for #
#=================================================================================#

def plot_front_title(data_obj, is_ansi=True):
    """ Create the title page for the pdf document, that states system info and the standard being used.

    :param data_obj: the object (Test_Object_Data or subclass) that holds all of the test result data
    :param is_ansi: whether or not the standard being tested against is ANSI N42.55 or ASTM F792
    """

    new_pdf_page(data_obj.pdf_obj, close_fig=False) #Create a new page
    plt.axis('off')

    #Determine which standard is being used, to choose which to include on the title page
    if is_ansi:
        test_std = "ANSI N42.55"
    else:
        test_std = "ASTM F792"

    # Display Title, that includes standard and x-ray information
    plt.text(0.15, 0.9, test_std + " test results for", ha='left', va='center', fontsize=20)
    date_str = (time.strftime("PDF generated %Y-%m-%d at %H:%M"))
    plt.text(0.15, 0.85, data_obj.xray_detector + ' system using', ha='left', va='center', fontsize=20)
    plt.text(0.15, 0.8, data_obj.xray_source + ' source', ha='left', va='center', fontsize=20)

    y0 = 0.8
    dy = 0.022
    x0 = 0.2

    # Display the comments, date, and info about code
    plt.text(x0, y0 - 3 * dy, data_obj.xray_comments, ha='left', va='center', fontsize=8)
    plt.text(x0, y0 - 4 * dy, date_str, ha='left', va='center', fontsize=8)
    plt.text(x0, y0 - 5 * dy, 'Analyzed using Glover ' + test_std + ' Python code (version 0.13)',
             ha='left', va='center', fontsize=8)


    #-------------------------------------------------------------------------------------#
    # Display all the images tested during the analysis, including the path to each image #
    #-------------------------------------------------------------------------------------#

    fs_files = 6
    ypos = y0 - 8 * dy
    dy = 0.019
    plt.text(x0, ypos, 'Main test object images:', ha='left', va='center', fontsize=8)
    ypos = ypos - dy
    for i, file in enumerate(data_obj.files):
        plt.text(x0, ypos, "     " + file, ha='left', va='center', fontsize=fs_files)
        ypos = ypos - dy

    # Include the Image extent if the standard being tested to is ANSI N42.55
    if is_ansi:
        ypos = ypos - 0.01
        plt.text(x0, ypos, 'Image extent image:', ha='left', va='center', fontsize=8)
        ypos = ypos - dy
        plt.text(x0, ypos, "    " + data_obj.extent_img_filename, ha='left', va='center', fontsize=fs_files)
        ypos = ypos - dy

    ypos = ypos - 0.01
    plt.text(x0, ypos, 'Noise images:', ha='left', va='center', fontsize=8)
    ypos = ypos - dy
    for i, file in enumerate(data_obj.noise_img_data):
        plt.text(x0, ypos, "    " + data_obj.noise_img_data[i].filename, ha='left', va='center', fontsize=fs_files)
        ypos = ypos - dy


def plot_overall_text(data_obj, yi, xpos, ha, va, fs, standard = "ANSI N42.55"):
    """ Create the main title and subtext for the results page.

    :param data_obj: the object (Test_Object_Data or subclass) that holds all of the test result data
    :param yi: the position on the y axis to plot the text to on the pdf page
    :param xpos: the position on the x axis to plot the text to
    :param ha: the position of the horizontal axis (center, left, etc)
    :param va: the position of the vertical axis
    :param fs: font size of the text
    :param standard: text for the standard being tested against (either ANSI N42.55 [default] or ASTM F792
    """

    new_pdf_page(data_obj.pdf_obj) #Start a new page
    plt.axis('off')

    plt.text(0.5, 0.9, standard + " test results", ha='center', va='center', fontsize=20)

    plt.text(xpos[0], yi, 'Metric name', ha=ha, va=va, fontsize=fs, weight='bold')
    plt.text(xpos[1], yi, 'Metric Value', ha=ha, va=va, fontsize=fs, weight='bold')
    plt.text(xpos[3], yi, 'Min. Performance Req.', ha=ha, va=va, fontsize=fs, weight='bold')


#-------------------------------------------------------------------------------------------------#
# NOTE: For the rest of these functions, the following is what is passed into the parameters -    #
#       if the function has these parameters.                                                     #
#                                                                                                 #
# data_obj: the object (Test_Object_Data or subclass) that holds all of the test result data      #
# test_num: the number for what test it is (in ascending order)                                   #
# yi:       the position on the y axis to plot the text to on the pdf page                        #
# xpos:     the position on the x axis to plot the text to                                        #
# ha:       the position of the horizontal axis (center, left, etc)                               #
# va:       the position of the vertical axis                                                     #
# fs:       font size of the text                                                                 #
# dfs:      the change in the font size (to shrink some of the text if need be)                   #
# standard: text for the standard being tested against (either ANSI N42.55 [default] or ASTM F792 #
#-------------------------------------------------------------------------------------------------#

def plot_pen_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall results text for the Penetration test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Penetration', ha=ha, va=va, fontsize=fs)
    plt.text(xpos[1], yi, str(data_obj.steel_pen) + ' mm$^\clubsuit$', ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\geq$ 6 mm', ha=ha, va=va, fontsize=fs - dfs)


def plot_BSNR_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall results text for the Organic Material Detection (BSNR) test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Organic Material Detection', ha=ha, va=va, fontsize=fs)
    str1 = "{0:.1f}".format(data_obj.POM_BSNR) + '$^\clubsuit$'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\geq$ 2.0', ha=ha, va=va, fontsize=fs - dfs)


def plot_spatial_text(data_obj, test_num, yi, yi2, xpos, ha, va, fs, dfs):
    """ Plot the overall results for the Spatial Resolution test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Spatial Resolution', ha=ha, va=va, fontsize=fs)
    plt.text(xpos[0], yi2, '  MTF20x', ha=ha, va=va, fontsize=fs - dfs)
    str1 = "{0:.2f}".format(data_obj.MTF_data.MTF20_x) + ' lp/mm'
    err = np.sqrt(data_obj.MTF_data.MTF20_x_err ** 2 + 0.0055 ** 2)
    str1 = str1 + '   $\pm$  ' + "{0:.2f}".format(err) + ' lp/mm$\dagger$'
    plt.text(xpos[1], yi2, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi2, '$\geq$ 0.5 lp/mm', ha=ha, va=va, fontsize=fs - dfs)


def plot_dyn_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall test results for the Dynamic Range test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Dynamic Range', ha=ha, va=va, fontsize=fs)
    str1 = "{0:.1f}".format(data_obj.dynamic_range)
    str1 = str1 + '   $\pm$  ' + "{0:.1f}".format(data_obj.dynamic_range_err) + '$\dagger$'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)

    plt.text(xpos[3], yi, '$\geq$ 150', ha=ha, va=va, fontsize=fs - dfs)


def plot_noise_text(data_obj, test_num, yi, dy, xpos, ha, va, fs, dfs):
    """ Plot the overall test results for the Noise test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Noise', ha=ha, va=va, fontsize=fs)

    yi = yi - dy[1]
    plt.text(xpos[0], yi, '  NEQx at 1 lp/mm', ha=ha, va=va, fontsize=fs - dfs)
    str1 = "{:,}".format(int(data_obj.NEQ_x_1))
    str1 = str1 + '   $\pm$  ' + "{:,}".format(int(data_obj.NEQ_x_1_err)) + '$\dagger$'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\geq$ 22,500', ha=ha, va=va, fontsize=fs - dfs)

    yi = yi - dy[1]
    plt.text(xpos[0], yi, '  NEQy at 1 lp/mm', ha=ha, va=va, fontsize=fs - dfs)
    str1 = "{:,}".format(int(data_obj.NEQ_y_1))
    str1 = str1 + '   $\pm$  ' + "{:,}".format(int(data_obj.NEQ_y_1_err)) + '$\dagger$'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\geq$ 22,500', ha=ha, va=va, fontsize=fs - dfs)


def plot_ff_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall results for the Flatness of Field test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Flatness of field ', ha=ha, va=va, fontsize=fs)
    str1 = "{0:.3f}".format(data_obj.field_flatness)
    str1 += '   $\pm$  ' + "{0:.2f}".format(data_obj.field_flatness_err) + '$\dagger$'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\geq$ 0.5', ha=ha, va=va, fontsize=fs - dfs)


def plot_extent_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall results for the Image extent test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Image extent', ha=ha, va=va, fontsize=fs)
    if data_obj.image_extent == None:
        str1 = 'None'
    else:
        str1 = "{0:.0f}".format(data_obj.image_extent) + ' mm' + '$^\clubsuit$'

    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\leq$ 10 mm', ha=ha, va=va, fontsize=fs - dfs)


def plot_area_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall results for the Image Area test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Image area', ha=ha, va=va, fontsize=fs)
    str1 = str(data_obj.image_area[0]) + ' by ' + str(data_obj.image_area[1]) + ' pixels'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\geq$ 1000 by 1000 pixels', ha=ha, va=va, fontsize=fs - dfs)


def plot_a_ratio_text(data_obj, test_num, yi, xpos, ha, va, fs, dfs):
    """ Plot the overall results for the Aspect Ratio test. """

    plt.text(xpos[0], yi, 'Test ' + str(test_num) + ': Aspect Ratio', ha=ha, va=va, fontsize=fs)
    str1 = "{0:.3f}".format(data_obj.aspect_ratio)
    str1 = str1 + '   $\pm$  ' + "{0:.3f}".format(np.abs(data_obj.aspect_ratio_err)) + '$\dagger$'
    plt.text(xpos[1], yi, str1, ha=ha, va=va, fontsize=fs - dfs)
    plt.text(xpos[3], yi, '$\leq$ 0.05', ha=ha, va=va, fontsize=fs - dfs)


def plot_overall_footnotes(xpos, ha, va, fs, dfs, standard="ANSI N42.55"):
    """ Add the footnotes to the overall results page. """

    #$\dagger$ is for the Spatial Resolution, Dynamic Range, Noise (NEQ), Flatness of Field, and Aspect Ratio
    y_legend = 0.2
    dy = 0.02
    plt.text(xpos[0],
             y_legend,
             '$\dagger$ These values represent the mean and one-sigma uncertainty in the quantity of interest. '
             + 'In some cases, ',
             ha=ha,
             va=va,
             fontsize=fs - dfs)

    y_legend -= dy
    plt.text(xpos[0],
             y_legend,
             '    the metric mean must be two sigma away from the min performance '
             + 'requirement in order to pass. See ',
             ha=ha,
             va=va,
             fontsize=fs - dfs)

    y_legend -= dy
    plt.text(xpos[0], y_legend, '    IEEE/' + standard + ' for full details.',
             ha=ha, va=va, fontsize=fs - dfs)

    #$\clubsuit$ - is for the (Useful) Penetration, Organic Material Detection, and Image Extent
    y_legend -= dy
    plt.text(xpos[0], y_legend, '$\clubsuit$ These tests do not have uncertainty values defined in the standard.',
             ha=ha, va=va, fontsize=fs - dfs)


#===================================================================================================#
# These functions are used for plotting the images used in the test, cropped and rotated along with #
# the different ROIs used in the test                                                               #
#===================================================================================================#
def plot_images(data_obj, fs):
    """ This function plots the images used in the analysis to the pdf object along with the important ROIs. """

    fig = new_pdf_page(data_obj.pdf_obj)        # Create a new page
    plt.suptitle('Cropped and Rotated Images')

    for i, file in enumerate(data_obj.files):

        # Specify the plot parameters
        ax2 = fig.add_subplot(3, 2, i + 1)
        plt.tick_params(axis='both', which='both', bottom='off', left='off',
                        top='off', right='off', labelbottom='off', labelleft='off')

        # Plot the image and the image title
        plt.imshow(data_obj.img_data[i].img, cmap=plt.get_cmap('gray'))
        plt.title('Orientation ' + str(data_obj.img_data[i].orientation), fontsize=fs)

        # Add the ROI rectangles to the plot
        data_obj.img_data[i].step_wedge_ROI.add_rect_to_plot(edgecolor='orange')
        data_obj.img_data[i].lead_foil_ROI.add_rect_to_plot(edgecolor='blue')
        data_obj.img_data[i].POM_piece_ROI.add_rect_to_plot(edgecolor='red')


def plot_image_footnotes(data_obj, xpos, ha, va, fs, dfs):
    """ Plot the footnotes to the cropped and rotated images - about the rotations and ROIs on the images. """

    # Setup the initial plotting values
    plt.subplot(3, 1, 3)
    plt.axis('off')
    y_legend = 0.8
    dy = 0.05

    #---------------------------------------------------------------------------------------------------#
    # Plot all the text of the footnotes. Use y_legend and dy to adjust the text position on the y axis #
    #---------------------------------------------------------------------------------------------------#
    plt.text(xpos[0],
             y_legend,
             "Image 1 is shown in its original orientation. The other three images have been rotated to be in the same orientation ",
             ha=ha,
             va=va,
             fontsize=fs - dfs)

    y_legend -= dy # Move the text directly below the previous text.
    plt.text(xpos[0],
             y_legend,
             'as image 1. Images 2, 3 and 4 were originally in an orientation that was rotated by 90, 180 and 270 degrees  ',
             ha=ha,
             va=va,
             fontsize=fs - dfs)

    y_legend -= dy
    plt.text(xpos[0],
             y_legend,
             'clockwise compared with image 1. This convention is continued throughout the document. ',
             ha=ha,
             va=va,
             fontsize=fs - dfs)

    y_legend -= dy
    y_legend -= dy # Leave a gap between the two "paragraphs"
    plt.text(xpos[0],
             y_legend,
             'Colored boxes have been drawn around important regions of the image. ',
             ha=ha,
             va=va,
             fontsize=fs - dfs)

    y_legend -= dy
    plt.text(xpos[0],
             y_legend,
             'The blue ROI should surround the Pb foil test piece, leaving some room on all sides of it.',
             ha=ha,
             va=va,
             fontsize=fs - dfs,
             color='blue')

    y_legend -= dy
    plt.text(xpos[0],
             y_legend,
             'The red ROI should be placed on the POM test piece, between the two bolts.',
             ha=ha,
             va=va,
             fontsize=fs - dfs,
             color='red')

    y_legend -= dy
    plt.text(xpos[0],
             y_legend,
             'The orange ROI should span the middle portion of the steel step wedge, running the entire vertical length of the image.',
             ha=ha,
             va=va,
             fontsize=fs - dfs,
             color='orange')



#========================================================================================#
# These functions are used to actually plot the test results of the processing to a pdf. #
#========================================================================================#

def plot_steel_pen_N4255(data_obj, test_num):
    """ This function plots the results for the Steel penetration test to the
    pdf object of the passed 'data_obj' for specifically N4255 images. """

    new_pdf_page(data_obj.pdf_obj) #Create a new page in the pdf for the test results

    #--------------------------------#
    # Display the title for the test #
    #--------------------------------#
    n_img = len(data_obj.img_data)
    plt.suptitle('Test ' + str(test_num) + ': Steel penetration  (' + str(data_obj.steel_pen) + ' mm)\n' +
                 'boundary images shown below')

    #------------------------------------------------------------------------------------#
    # Display the boundaries of the test for the steps in the steel wedge for each image #
    #------------------------------------------------------------------------------------#
    ROI_size = [15 / data_obj.px_size[0], 5 / data_obj.px_size[1]]

    # Loop for each boundary (step) in the steel wedge / test
    for i_boundary in np.arange(12):

        # Go through each image and display the individual boundaries for the penetration test
        for i_img in np.arange(n_img):

            # ax2 = fig.add_subplot(n_img,12,12*i_img + i_boundary + 1)
            plt.subplot(n_img, 12, 12 * i_img + i_boundary + 1)

            plt.tick_params(axis='both', which='both', bottom='off', left='off', top='off', right='off',
                            labelbottom='off', labelleft='off')

            #-------------------------------------------------------#
            # Create a new ROI for each "boundary slice" to display #
            #-------------------------------------------------------#
            center0 = data_obj.img_data[i_img].step_wedge_boundaries[i_boundary]
            center1 = data_obj.img_data[i_img].step_wedge_ROI.shape[1] / 2

            ROI_i = ROI(data_obj.img_data[i_img].step_wedge_ROI.get_img(), (center0, center1), ROI_size)

            # Calculate the boundary signal for the ROI
            bs = calc_boundary_sig(ROI_i, data_obj.img_data[i_img].px_size,
                                               ROI_size_mm=2, plot=True, force_middle=True)

            #------------------------------------------------------------------------------#
            # Display the additional details to the pdf, formatted for location in the pdf #
            #------------------------------------------------------------------------------#

            # Display the area (in mm) that the ROI covers
            if i_img == 0:
                plt.title(str(3 * i_boundary) + '/' + str(3 * (i_boundary + 1)) + ' mm', fontsize=7)

            #Display the BSNR values for the test
            elif i_img == 3:
                plt.title('BSNR=' + "{0:.1f}".format(data_obj.steel_pen_BSNR[i_boundary]), fontsize=6)

            # Display the image (orientation) number
            if i_boundary == 0:
                plt.ylabel('Orientation ' + str(i_img + 1), fontsize=7)


def plot_BSNR(data_obj, test_num, cmap):
    """ This function plots the results of the BSNR (Organic Material Detection) test to the
    pdf object of the passed 'data_obj' """

    #----------------------------------------------------------------#
    # Create a new page in the pdf and display the title of the test #
    #----------------------------------------------------------------#
    fig = new_pdf_page(data_obj.pdf_obj)
    plt.axis('off')
    plt.suptitle('Test ' + str(test_num) + ': Organic Detection  (' + "{0:.1f}".format(data_obj.POM_BSNR) + ')')

    #------------------------------------------------------------------------------------------#
    # Display the statistics of the test - BSNR result, average signal, and standard deviation #
    #------------------------------------------------------------------------------------------#
    str1 = 'BSNR: ' + "{0:.1f}".format(data_obj.POM_BSNR)
    plt.text(0.5, 0.95, str1, ha='center', fontsize=8)

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout

    str2 = 'Ave sig ' + "{0:.3f}".format(np.average(data_obj.POM_boundary_signal_arr))
    plt.text(0.5, 0.90, str2, ha='center', fontsize=8)
    str3 = 'Stdev ' + "{0:.3f}".format(np.std(data_obj.POM_boundary_signal_arr, ddof=1))
    plt.text(0.5, 0.88, str3, ha='center', fontsize=8)

    #----------------------------------------------------------------------------------------#
    # Go through each image [file = image?] and plot the POM ROI images with their sub ROI's #
    #----------------------------------------------------------------------------------------#

    n_files = len(data_obj.files)
    for i in range(n_files):

        # Set up the design of the page - rows / columns, subplots, etc.
        row_col = np.ceil(np.sqrt(n_files * 1.0))
        ax1 = fig.add_subplot(int(row_col * 100 + row_col * 10 + 1 + i))
        plt.axis('off')

        # Display the ROI image of the "POM_piece", making sure to specify the range of
        # values for the colormap
        vmin = data_obj.img_data[i].POM_piece_ROI.img.min()
        vmax = np.percentile(data_obj.img_data[i].POM_piece_ROI.img.ravel(), 99.5)
        plt.imshow(data_obj.img_data[i].POM_piece_ROI.img, cmap=cmap, vmin=vmin, vmax=vmax)

        # Add each sub ROI and plot them to the pdf - specifically to the current image
        for subROI in data_obj.img_data[i].POM_piece_ROI.subROIs:
            subROI.add_rect_to_plot()

        # Display the text for the Boundary Signal for each image
        str1 = 'Bound. sig.=' + "{0:.3f}".format(data_obj.img_data[i].POM_boundary_signal)
        plt.title(str1, fontsize=7)


def plot_spatial_res(data_obj, test_num):
    """ Plots the results of the Spatial Resolution test to the pdf object of the passed 'data_obj'"""

    #---------------------------------------------------------------------------------------------#
    # Go through each file [image] and plot the spatial resolution ROI images and MTF data graphs #
    #---------------------------------------------------------------------------------------------#
    for i in range(len(data_obj.files)):

        fig = new_pdf_page(data_obj.pdf_obj) #Create a new pdf page
        plt.axis('off')

        # Create and plot the (centered) title of the page - that also states the MTF20 values
        str1 = 'Test ' + str(test_num) + ': Spatial Resolution \n' + \
               'MTF20x = ' + "{0:.2f}".format(data_obj.MTF_data.MTF20_x) + \
               '\n MTF20y = ' + "{0:.2f}".format(data_obj.MTF_data.MTF20_y)
        plt.suptitle(str1)

        #-----------------------------------------------#
        # Plot the main image in the middle of the page #
        #-----------------------------------------------#
        ax1 = fig.add_subplot(312)
        plt.imshow(data_obj.img_data[i].lead_foil_ROI.img) #Display the ROI image of the lead foil

        # Display the title of the ROI image - the orientation
        plt.title('Orientation ' + str(data_obj.img_data[i].orientation))
        plt.xticks([])  # labels
        plt.yticks([])
        ax = plt.gca()
        ax.xaxis.set_ticks_position('none')  # tick markers
        ax.yaxis.set_ticks_position('none')

        if data_obj.img_data[i].orientation % 2 == 1:
            plt.xlabel('x axis')
            plt.ylabel('y axis')
        else:
            plt.xlabel('y axis')
            plt.ylabel('x axis')

        #-----------------------------#
        # Display the Horizontal plot #
        #-----------------------------#
        ax2 = fig.add_subplot(325)
        plt.ylabel('MTFx')
        plt.xlabel('Spatial frequency (cycles/mm)')
        plt.plot(data_obj.img_data[i].MTF_obj.MTF_x_f, data_obj.img_data[i].MTF_obj.MTF_x)
        text = 'MTF20x = ' + "{0:.2f}".format(data_obj.img_data[i].MTF_obj.MTF20_x)

        ax2.annotate(text,
                     xy=(data_obj.img_data[i].MTF_obj.MTF20_x, 0.2),
                     xytext=(np.max(data_obj.img_data[i].MTF_obj.MTF_x_f), 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                     horizontalalignment='right')

        #---------------#
        # Vertical plot #
        #---------------#
        ax3 = fig.add_subplot(326)
        plt.ylabel('MTFy')
        plt.xlabel('Spatial frequency (cycles/mm)')
        plt.plot(data_obj.img_data[i].MTF_obj.MTF_y_f, data_obj.img_data[i].MTF_obj.MTF_y)
        text = 'MTF20y = ' + "{0:.2f}".format(data_obj.img_data[i].MTF_obj.MTF20_y)

        ax3.annotate(text,
                     xy=(data_obj.img_data[i].MTF_obj.MTF20_y, 0.2),
                     xytext=(np.max(data_obj.img_data[i].MTF_obj.MTF_y_f), 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                     horizontalalignment='right')


def plot_dynamic_range(data_obj, test_num):
    """ Plot the results of the Dynamic Range test to the pdf object of the passed 'data_obj'"""

    #----------------------------------------------------------------#
    # Create a new page in the pdf and display the title of the test #
    #----------------------------------------------------------------#
    fig = new_pdf_page(data_obj.pdf_obj)
    plt.suptitle('Test ' + str(test_num) + ': Dynamic Range  (' + "{0:.1f}".format(data_obj.dynamic_range) + ')')

    #-----------------------------------------------------------------#
    # Go through each file [think image] and display the entire image #
    # along with information of the dynamic range (bright vs dark)    #
    #-----------------------------------------------------------------#
    for i in range(len(data_obj.files)):

        # Create subplots and set up
        ax2 = fig.add_subplot(2, 2, i + 1)
        plt.tick_params(axis='both', which='both', bottom='off', left='off', top='off', right='off',
                        labelbottom='off', labelleft='off')

        #---------------------------------------------------------------------------------------------#
        # Plot the image to the pdf along with the values of the bright average and dark standard dev #
        #---------------------------------------------------------------------------------------------#
        plt.imshow(data_obj.img_data[i].img, cmap=plt.get_cmap('gray'))
        plt.title('Orientation ' + str(data_obj.img_data[i].orientation) + '\n' +
                  'dark stdev:' + "{0:.1f}".format(data_obj.img_data[i].dynamic_range_dark_ROI.get_stdev()) + '\n' +
                  'bright ave:' + "{0:.0f}".format(data_obj.img_data[i].dynamic_range_light_ROI.get_ave()),
                  fontsize=7)

        #------------------------------------------------------------------------------------------#
        # Add the ROIs of the dynamic range on top of the image and create arrows pointing to them #
        #------------------------------------------------------------------------------------------#
        data_obj.img_data[i].dynamic_range_light_ROI.add_rect_to_plot()
        data_obj.img_data[i].dynamic_range_dark_ROI.add_rect_to_plot()

        ax = plt.gca()

        # The arrow with text for the bright ROI
        xy = (data_obj.img_data[i].dynamic_range_light_ROI.center)[::-1]
        ax.annotate('bright ROI', xy=xy, color='green',
                    xycoords='data', xytext=(0.1, 0.99), textcoords='axes fraction',
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    horizontalalignment='left', verticalalignment='top',
                    )

        # The arrow and text for the dark ROI
        xy = (data_obj.img_data[i].dynamic_range_dark_ROI.center)[::-1]
        ax.annotate('Dark ROI', xy=xy, color='green',
                    xycoords='data', xytext=(0.9, 0.05), textcoords='axes fraction',
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )


def plot_noise(data_obj, test_num):
    """ Plot the results of the Noise test to the pdf object of the passed 'data_obj' """

    #=======================================================================#
    # Create and plot the page for the X values of the test (NEQ, NPS, MTF) #
    #=======================================================================#

    fig = new_pdf_page(data_obj.pdf_obj) #Create a new page in the pdf

    # Display the (centered) title of the figures / page
    plt.suptitle('Test ' + str(test_num) +
                 ': Noise  ($NEQ_x$ at 1 lp/mm: ' +
                 "{:,}".format(int(data_obj.NEQ_x_1)) + ')')

    #--------------------------------------------------------------------------#
    # Display the equation and variables of the test to the middle of the page #
    #--------------------------------------------------------------------------#
    fs3 = 10
    plt.subplot(3, 1, 1)
    plt.axis('off')

    # Display the main equation for calculating the NEQ
    str1 = "$ NEQ = \\frac{S_{out}^2 MTF^2}{NPS} $"
    plt.text(0.3, 0.6, str1, ha='left', fontsize=16)

    # Display the values of the different variables in the NEQ equation
    str1 = "$NPS_x$ (at 1 lp/mm) = " + "{0:.1f}".format((data_obj.NPS_x_1))
    plt.text(0.3, 0.4, str1, ha='left', fontsize=fs3)

    str1 = "$MTF_x$ (at 1 lp/mm) = " + "{0:.3f}".format((data_obj.MTF_x_1))
    plt.text(0.3, 0.3, str1, ha='left', fontsize=fs3)

    str1 = "$S_{out}$ (at 1 lp/mm) = " + "{:,}".format(int(data_obj.S_out))
    plt.text(0.3, 0.2, str1, ha='left', fontsize=fs3)

    str1 = "$NEQ_x$ (at 1 lp/mm) = " + "{:,}".format(int(data_obj.NEQ_x_1))
    str1 = str1 + '   $\pm$  ' + "{:,}".format(int(data_obj.NEQ_x_1_err))
    plt.text(0.3, 0.1, str1, ha='left', fontsize=fs3)


    #------------------------------------------------#
    # Display the graphs for NEQ_x, MTF_x, and NPS_x #
    #------------------------------------------------#

    plt.subplot(3, 2, 3)
    plt.ylabel('$MTF_x$')
    plt.xlabel('Line-pairs per mm')
    plt.plot(data_obj.MTF_data.MTF_x_f, data_obj.MTF_data.MTF_x)

    plt.subplot(3, 2, 4)
    plt.ylabel('$\\sqrt{NPS_x}$')
    plt.xlabel('Line-pairs per mm')
    plt.plot(data_obj.MTF_data.MTF_x_f[1:], np.sqrt(data_obj.NPS_x_interp[1:]))

    plt.subplot(3, 1, 3)
    plt.ylabel('$\\sqrt{NEQ_x}$')
    plt.xlabel('Line-pairs per mm')
    plt.plot(data_obj.MTF_data.MTF_x_f, np.sqrt(data_obj.NEQ_x))

    ind = int(1 / 0.02)
    plt.plot(data_obj.MTF_data.MTF_x_f[ind], np.sqrt(data_obj.NEQ_x[ind]), 'ro')
    plt.tight_layout()


    ##=======================================================================#
    # Create and plot the page for the X values of the test (NEQ, NPS, MTF) #
    #=======================================================================#

    fig = new_pdf_page(data_obj.pdf_obj) #Create a new page in the pdf

    # Display the (centered) title of the figures / page
    plt.suptitle('Test ' + str(test_num) +
                 ': Noise  ($NEQ_y$ at 1 lp/mm: ' +
                 "{:,}".format(int(data_obj.NEQ_y_1)) + ')')

    # --------------------------------------------------------------------------#
    # Display the equation and variables of the test to the middle of the page #
    # --------------------------------------------------------------------------#
    fs3 = 10
    plt.subplot(3, 1, 1)
    plt.axis('off')

    # Display the main equation for calculating the NEQ
    str1 = "$ NEQ = \\frac{S_{out}^2 MTF^2}{NPS} $"
    plt.text(0.3, 0.6, str1, ha='left', fontsize=16)

    # Display the values of the different variables in the NEQ equation
    str1 = "$NPS_y$ (at 1 lp/mm) = " + "{0:.1f}".format((data_obj.NPS_y_1))
    plt.text(0.3, 0.4, str1, ha='left', fontsize=fs3)

    str1 = "$MTF_y$ (at 1 lp/mm) = " + "{0:.3f}".format((data_obj.MTF_y_1))
    plt.text(0.3, 0.3, str1, ha='left', fontsize=fs3)

    str1 = "$S_{out}$ (at 1 lp/mm) = " + "{:,}".format(int(data_obj.S_out))
    plt.text(0.3, 0.2, str1, ha='left', fontsize=fs3)

    str1 = "$NEQ_y$ (at 1 lp/mm) = " + "{:,}".format(int(data_obj.NEQ_y_1))
    str1 = str1 + '   $\pm$  ' + "{:,}".format(int(data_obj.NEQ_y_1_err))
    plt.text(0.3, 0.1, str1, ha='left', fontsize=fs3)

    #------------------------------------------------#
    # Display the graphs for NEQ_y, MTF_y, and NPS_y #
    #------------------------------------------------#

    plt.subplot(3, 2, 3)
    plt.ylabel('$MTF_y$')
    plt.xlabel('Line-pairs per mm')
    plt.plot(data_obj.MTF_data.MTF_y_f, data_obj.MTF_data.MTF_y)

    plt.subplot(3, 2, 4)
    plt.ylabel('$\\sqrt{NPS_y}$')
    plt.xlabel('Line-pairs per mm')
    plt.plot(data_obj.MTF_data.MTF_y_f[1:], np.sqrt(data_obj.NPS_y_interp[1:]))

    plt.subplot(3, 1, 3)
    plt.ylabel('$\\sqrt{NEQ_y}$')
    plt.xlabel('Line-pairs per mm')
    plt.plot(data_obj.MTF_data.MTF_y_f, np.sqrt(data_obj.NEQ_y))

    ind = int(1 / 0.02)
    plt.plot(data_obj.MTF_data.MTF_y_f[ind], np.sqrt(data_obj.NEQ_y[ind]), 'ro')
    plt.tight_layout()



def plot_field_flatness(data_obj, test_num):
    """ Plot the results of the Field of Flatness test to the pdf object of the passed 'data_obj' """

    # Create a new page in the pdf and display the title of the test
    fig = new_pdf_page(data_obj.pdf_obj)
    plt.suptitle('Test ' + str(test_num) + ': Flatness of Field  (' + "{0:.2f}".format(data_obj.field_flatness) + ')')

    # Setup some initial variables for the text: font size, color, and stroke
    fs = 9
    color = 'white'
    path_effects = [PathEffects.withStroke(linewidth=2, foreground="black")]

    #--------------------------------------------------------------------------#
    # Go through each noise image and display the ROI of the flatness of field #
    # test along with the data from the test                                   #
    #--------------------------------------------------------------------------#
    for i in range(len(data_obj.noise_img_data)):

        # Setup the variables for aligning the result text within the image
        xpos = 0.1 * data_obj.noise_img_data[i].flatness_ROI.shape[1]
        ypos = 0.0 * data_obj.noise_img_data[i].flatness_ROI.shape[0]
        sep = 0.08 * data_obj.noise_img_data[i].flatness_ROI.shape[0]

        # Display the ROI of the noise image
        plt.subplot(2, 3, i + 1)
        plt.imshow(data_obj.noise_img_data[i].flatness_ROI.img, cmap='Greys_r')

        #---------------------------------------------------------------------------------#
        # Get the name of the image file from the file path, and set it as the plot title #
        #---------------------------------------------------------------------------------#
        file_short = data_obj.noise_img_data[i].filename
        file_short = file_short.replace('\\', ' ')
        file_short = file_short.replace('/', ' ')
        file_short = file_short.split()[-1]

        plt.title(file_short, fontsize=fs - 1)
        plt.tick_params(labelsize=6)

        #----------------------------------------------------------------------------------------------------#
        # Display the results of the flatness of field test: standard deviation, mean, and flatness of field #
        #----------------------------------------------------------------------------------------------------#
        plt.text(xpos, ypos + 3 * sep,
                 "stdev: " + "{0:.0f}".format(data_obj.noise_img_data[i].flatness_ROI.get_stdev()),
                 ha='left', va='top', fontsize=fs, color=color, path_effects=path_effects)
        plt.text(xpos, ypos + 4 * sep, "mean: " + "{0:.0f}".format(data_obj.noise_img_data[i].flatness_ROI.get_ave()),
                 ha='left', va='top', fontsize=fs, color=color, path_effects=path_effects)
        plt.text(xpos, ypos + 2 * sep, "flatness of field: " + "{0:.2f}".format(data_obj.noise_img_data[i].flatness),
                 ha='left', va='top', fontsize=fs, color=color, path_effects=path_effects)

        #-----------------------------------------------------------------#
        # Display additional text if the coverage of the ROI is too small #
        #-----------------------------------------------------------------#
        cov = data_obj.noise_img_data[i]._cov_frac

        if np.min(cov) < 0.99:
            plt.text(xpos, ypos + 7 * sep, "ROI too small", ha='left', va='top', fontsize=fs, color=color,
                     path_effects=path_effects)
            plt.text(xpos, ypos + 8 * sep, "coverage frac: " + "{0:.2f}".format(cov[0]) + ' '"{0:.2f}".format(cov[1]),
                     ha='left', va='top', fontsize=fs, color=color, path_effects=path_effects)

    plt.tight_layout()



def plot_image_extent(data_obj, test_num):
    """ Plot the results of the Image Extent test to the pdf object of the passed 'data_obj' """

    fig = new_pdf_page(data_obj.pdf_obj) # Create a new page in the pdf

    # Display information about the image extent if it exists
    if data_obj.image_extent is not None:
        plt.suptitle('Test ' + str(test_num) + ': Image Extent (' + "{0:.1f}".format(data_obj.image_extent) + ' mm)')
    else:
        plt.suptitle('Test ' + str(test_num) + ': Image Extent (None)')


    #----------------------------------------------------------------------------------------------#
    # Display the extent image along with the ROIs of the dynamic range for the image, with arrows #
    # pointing to them                                                                             #
    #----------------------------------------------------------------------------------------------#
    plt.subplot(212)
    plt.imshow(data_obj.extent_img, cmap=plt.get_cmap('gray'))
    data_obj.extent_dark_ROI.add_rect_to_plot()
    data_obj.extent_bright_ROI.add_rect_to_plot()

    # Create the arrows
    arrowprops = dict(facecolor='green', shrink=0.05)
    plt.annotate('bright spot', xy=(data_obj.extent_bright_ROI.center[::-1]), xytext=(0.89, 0.95),
                 arrowprops=arrowprops, color='green', textcoords='axes fraction',
                 horizontalalignment='right', verticalalignment='top')

    plt.annotate('dark spot', xy=(data_obj.extent_dark_ROI.center[::-1]), xytext=(0.98, 0.8),
                 arrowprops=arrowprops, color='green', textcoords='axes fraction',
                 horizontalalignment='right', verticalalignment='top')

    plt.subplot(211)
    plt.axis('off')

    #--------------------------------------------------------------------------------------------------#
    # For each distance given (in mm), go through and display the dynamic range for the given distance #
    #--------------------------------------------------------------------------------------------------#
    fontsize2 = 8
    step = 0.05
    dist = (1, 5, 10)

    for i in range(len(dist)):
        plt.text(0.3,
                 0.5 - i * step,
                 'distance '
                 + str(dist[i])
                 + ' mm '
                 + '    dynamic range '
                 + "{0:.1f}".format(data_obj.extent_dyn_range[i]),
                 ha='left',
                 va='center',
                 fontsize=fontsize2)


def plot_aspect_ratio(data_obj, test_num, cmap, aspect_corr):
    """ Plot the results of the Aspect Ratio test to the pdf object of the passed 'data_obj' """

    # Create a new page in the pdf and display the title for the aspect ratio test, including the number result
    fig = new_pdf_page(data_obj.pdf_obj)
    plt.suptitle('Test ' + str(test_num) + ': Aspect Ratio  (' + "{0:.3f}".format(data_obj.aspect_ratio) + ')')

    #-------------------------------------------------------#
    # ADD NOTE OF WHAT IS HAPPENING OVERALL IN THE FOR LOOP #
    #-------------------------------------------------------#
    for i in range(len(data_obj.img_data)):
        plt.subplot(5, 3, i * 3 + 4)
        # plt.axis('off')

        # Create and display the title of the graph
        str1 = "Image " + str(data_obj.img_data[i].orientation) + " Aspect Ratio: " + \
               "{0:.3f}".format(data_obj.img_data[i].aspect_ratio)
        plt.title(str1, fontsize=10)

        # Display the ROI of the foil and setup the plot
        plt.imshow(data_obj.img_data[i].lead_foil_ROI.img, cmap=cmap)
        plt.xticks([])  # labels
        plt.yticks([])
        ax = plt.gca()
        ax.xaxis.set_ticks_position('none')  # tick markers
        ax.yaxis.set_ticks_position('none')

        # If the orientation number is odd, make sure to swap the axis so it lines up correctly
        # (as it is "on its side" / rotated 90 degrees
        if data_obj.img_data[i].orientation % 2 == 1:
            plt.xlabel('x axis')
            plt.ylabel('y axis')
            c1 = 'b'
            c2 = 'r'
        else:
            plt.xlabel('y axis')
            plt.ylabel('x axis')
            c1 = 'r'
            c2 = 'b'

        # Create the blue and red lines through the middle of the ROI on each axes
        shape = data_obj.img_data[i].lead_foil_ROI.shape
        plt.plot((shape[0] / 2, shape[0] / 2), (0, shape[1]), color=c2)
        plt.plot((0, shape[0]), (shape[1] / 2, shape[1] / 2), color=c1)

        # Plot the graphs of the aspect ratio data using the N4255_Image method
        data_obj.img_data[i].measure_aspect_ratio_from_foil(plot=True,
                                                            subplot=(5, 3, i * 3 + 5),
                                                            aspect_corr=aspect_corr)

    plt.tight_layout()

