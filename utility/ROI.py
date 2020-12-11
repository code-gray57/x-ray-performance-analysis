import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

class ROI:
    """ This class represents a Region of Interest within a given image.
        The class holds the center, size, shape, and coordinates of the ROI, along with
        the actual image area that the ROI specifies and any sub ROI's within the ROI. """

    def __init__(self, img, center, size, name='', store_img=True):

        # Initialize the boundaries of the ROI - center, coords, size, etc.
        self.center = center
        self.size = size
        self.shape = size
        self.name = name
        
        self.coords  = np.array([int(center[0]-size[0]/2), int(center[0]+size[0]/2), \
                                 int(center[1]-size[1]/2), int(center[1]+size[1]/2)])

        # Set up the attributes for average and standard deviation, for future calculations
        self.stdev = None
        self.ave = None

        # Used to hold the image section dictated by the ROI and its shape - if store_img flag is true
        self.img = None
        self.img_shape = img.shape
        
        # yep, I did it ROIception
        self.subROIs = []  # A list of ROI's within the ROI this object represents

        # Error-check the coordinates to make sure they do not go beyond the boundaries of the image
        if min(self.coords) < 0:
            self.coords[self.coords < 0] = 0
            print('ROI ' + self.name + ' boundary outside image')
        if self.coords[1] > img.shape[0]:
            self.coords[1] = img.shape[0]
            print('ROI ' + self.name + ' bottom boundary outside image bottom')
        if self.coords[3] > img.shape[1]:
            self.coords[3] = img.shape[1]
            print('ROI ' + self.name + ' right boundary outside image right')

        # Save the section of the image dictated by the ROI coordinates
        if store_img:
            self.img = img[self.coords[0]:self.coords[1], self.coords[2]:self.coords[3]]

    
    def get_stdev(self, recalc=False):
        """ Returns the ROI's standard deviation. If recalc = True, will recalculate the value before returning it. """
        if self.stdev is not None and not recalc:
            return self.stdev

        self.stdev = np.std(self.img.ravel())
        return self.stdev
    
    def get_ave(self, recalc=False):
        """ Returns the ROI's average. If recalc = True, will recalculate the value before returning it. """
        if self.ave is not None and not recalc:
            return self.ave

        self.ave = np.mean(self.img.ravel())
        return self.ave
    
    def get_edge_coords(self):
        """ Returns the coordinates of ROI's dimensions. """
        return self.coords
    
    def get_img(self, img=None):
        """ Returns the image that the ROI encapsulates. If there is no image but an image is passed as
        an argument, then the area of that image that the ROI covers is returned. """

        if self.img is None: #No image specified to the ROI object

            # If no image is saved, check if an image was passed. If so, return the ROI of that image.
            if img is None:
                print('no image provided')
            else:
                return img[self.coords[0]:self.coords[1], self.coords[2]:self.coords[3]]
        else:
            return self.img
        
    
    def add_rect_to_plot(self, edgecolor='g', linewidth=1, facecolor='none', label=None, use_ax=None):
        """ This method adds the area specified by the ROI object to the current plot as a rectangle.
            The color, width, and other specific details of the rectangle are passed as arguments. """

        # Create the anchor point for the rectangle and create the Rectangle object
        coord = (self.center[1] - self.size[1] / 2, self.center[0] - self.size[0] / 2)

        rect = patches.Rectangle(coord, self.size[1], self.size[0], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor=facecolor)

        # If a new axis should be used to plot the rectangle, create a new one
        if use_ax is None:
            use_ax = plt.gca()

        use_ax.add_patch(rect) # Add the rectangle to the axis

        # Create a text label for the rectangle if the text is specified.
        if label is not None:
            use_ax.text(self.center[1], self.center[0], label, color=edgecolor,
                     verticalalignment='center', horizontalalignment='center',)


    def add_subROI(self, ROI_in):
        """ Add an ROI to the list of sub ROI's. """
        self.subROIs.append(ROI_in)


    # We must undo rot90 anticlockwise rotations and a flip_lr
    def reverse_transformation(self, rot90, flip_lr=False):
        """ Undo any 90 degree rotation and/or left-right flip and return a copy ROI with the changes. """

        transformed_ROI = copy.deepcopy(self) #Make a copy of itself to transform and return

        # Undo the left-right flip
        if flip_lr:
            transformed_ROI.center = (transformed_ROI.center[0], self.img_shape[1] - transformed_ROI.center[1])

        #Undo the rotation of the image
        cent = transformed_ROI.center
        if rot90 == 1:
            transformed_ROI.center = (cent[1], self.img_shape[0] - cent[0])
            transformed_ROI.size   = (transformed_ROI.size[1], transformed_ROI.size[0])
        elif rot90 == 2:
            transformed_ROI.center = (self.img_shape[0] - cent[0], self.img_shape[1] - cent[1])
        elif rot90 == 3:
            transformed_ROI.center = (self.img_shape[1] - cent[1], cent[0])
            transformed_ROI.size = (transformed_ROI.size[1], transformed_ROI.size[0])

        return transformed_ROI

    