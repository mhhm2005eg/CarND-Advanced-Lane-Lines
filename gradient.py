import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import *


threshold = {'x':(20, 100), 'y':(20, 100) ,'m':(30, 100) , 'd':(0.7, 1.3)}

def canny_test():
    #fig = plt.figure()
    # Read in the image and convert to grayscale
    image = mpimg.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if False:
        # Define a kernel size for Gaussian smoothing / blurring
        # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
        kernel_size = 7#15 #has to be odd
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    else:
        blur_gray = gray

    # Define parameters for Canny and run it
    # NOTE: if you try running this code you might want to change these!
    low_threshold = 50#50
    high_threshold = 100#110
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    f0 = plt.figure(0)
    plt.imshow(edges, cmap='Greys_r')
    plt.title("canny")
    f0.show()
    # Display the image
    #fig1 = plt.figure(1)
    #plt.imshow(gray, cmap='Greys_r')

    #fig2 = plt.figure(2)
    #plt.imshow(blur_gray, cmap='Greys_r')
    #fig1.show()
    #plt.show()
    if False:
        fig3 = plt.figure(3)
        for i in range(1,50,5):
            low_threshold = i  # 50
            high_threshold = 100  # 110
            edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
            plt.imshow(edges, cmap='Greys_r')
            #fig2.show()
            plt.show()
    plt.show()

def sobel(gray, x, y, thresh_min = 20, thresh_max = 100):
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return scaled_sobel, sbinary

def sobel_filter():
    image = mpimg.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    scaled_sobel, sbinary = sobel(gray, 0, 1)
    f0 = plt.figure(0)
    plt.imshow(scaled_sobel, cmap='gray')
    plt.title("scaled_sobel")
    f0.show()
    f1 =plt.figure(1)
    plt.imshow(sbinary, cmap='gray')
    plt.title("sbinary")
    f1.show()


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=threshold['x']):
    orientation ={'x':(1, 0), 'y':(0, 1)}
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F,*orientation[orient], ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=threshold['m']):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img)  # Remove this line
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    #mag, ang = cv2.cartToPolar(sobelx, sobely)
    sobel_m = np.sqrt(np.square(sobelx)+np.square(sobely))
    sobelm = np.uint8(255 * sobel_m / np.max(sobel_m))
    binary_output = np.zeros_like(sobelm)
    binary_output[(sobelm >= mag_thresh[0]) & (sobelm <= mag_thresh[1])] = 1
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=threshold['d']):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img)  # Remove this line
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))

    sobel_slope = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(sobel_slope)
    binary_output[(sobel_slope >= thresh[0]) & (sobel_slope <= thresh[1])] = 1

    return binary_output



def main1():
    #canny_test()
    #sobel_filter()
    fig0 =plt.figure(0)
    B = mag_thresh(image, sobel_kernel=3,  mag_thresh=(30, 100))
    plt.imshow(B, cmap='gray')
    fig1 = plt.figure(1)
    B = mag_thresh(image, sobel_kernel=9,  mag_thresh=(30, 100))
    plt.imshow(B, cmap='gray')
    fig2 = plt.figure(2)
    B = mag_thresh(image, sobel_kernel=13,  mag_thresh=(30, 100))
    plt.imshow(B, cmap='gray')

    fig3 = plt.figure(3)
    B = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    plt.imshow(B, cmap='gray')

    plt.show()
    #input("Press Enter to continue...")

def main():
    imgs =[["gradx", "grady", "mag_binary"], ["dir_binary", "combined", "combined1"]]
    # Choose a Sobel kernel size
    ksize = 15  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined1 = np.zeros_like(dir_binary)
    combined1[((gradx == 1) & (dir_binary == 1))] = 1

    col_no = 3
    raw_no = 2
    f, ax = plt.subplots(raw_no, col_no, figsize=(24, 9))
    f.tight_layout()
    for c in range(0,col_no):
        for r in range(0, raw_no):
            #plt.imshow(exec(im), cmap='gray')
            ax[r][c].imshow(eval(imgs[r][c]), cmap='gray')
            ax[r][c].set_title(imgs[r][c], fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    img_file = 'test_images/signs_vehicles_xygrad.png'
    image = mpimg.imread(img_file)

    main()
    pass
