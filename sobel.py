import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import *


class sobel:
    def __init__(self, caller=None):
        self.threshold = {'x':(20, 100), 'y':(20, 100) ,'m':(30, 100) , 'd':(0.7, 1.3)}
        self.caller = caller
        self.image = caller.image
        self.sobel_kernel = 3


    def abs_sobel_thresh(self, orient='x', sobel_kernel=None, thresh=None, masked=False):
        orientation ={'x':(1, 0), 'y':(0, 1)}
        if thresh is None:
            thresh = self.threshold[orient]
        if  sobel_kernel is None:
            sobel_kernel = self.sobel_kernel

        gray = self.caller.gray.get()
        sobel = cv2.Sobel(gray, cv2.CV_64F,*orientation[orient], ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Calculate directional gradient
        # Apply threshold
        if masked:
            grad_binary = self.caller.region_of_interest(grad_binary)
        return grad_binary

    def mag_thresh(self, sobel_kernel=None, mag_thresh=None, masked=False ):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Calculate the magnitude
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        # 5) Create a binary mask where mag thresholds are met
        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img)  # Remove this line
        if mag_thresh is None:
            mag_thresh = self.threshold['m']
        if  sobel_kernel is None:
            sobel_kernel = self.sobel_kernel

        gray = self.caller.gray.get()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
        #mag, ang = cv2.cartToPolar(sobelx, sobely)
        sobel_m = np.sqrt(np.square(sobelx)+np.square(sobely))
        sobelm = np.uint8(255 * sobel_m / np.max(sobel_m))
        binary_output = np.zeros_like(sobelm)
        binary_output[(sobelm >= mag_thresh[0]) & (sobelm <= mag_thresh[1])] = 1

        if masked:
            binary_output = self.caller.region_of_interest(binary_output)

        return binary_output


    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, sobel_kernel=None, thresh=None, masked=False):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        # 5) Create a binary mask where direction thresholds are met
        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img)  # Remove this line

        if thresh is None:
            thresh = self.threshold['d']
        if  sobel_kernel is None:
            sobel_kernel = self.sobel_kernel


        gray = self.caller.gray.get()
        abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))

        sobel_slope = np.arctan2(abs_sobely, abs_sobelx)

        binary_output = np.zeros_like(sobel_slope)
        binary_output[(sobel_slope >= thresh[0]) & (sobel_slope <= thresh[1])] = 1
        if masked:
            binary_output = self.caller.region_of_interest(binary_output)

        return binary_output

