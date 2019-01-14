import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from common import *
from common import hist
from calibration import load_calib
from gradient import abs_sobel_thresh, mag_thresh, dir_threshold
from color_space import color_select, draw_sub_plots, stack_binary_images
from perspective_transformation import get_source_dist_points, get_source_dist_points_test, unwrap

img_file = 'bridge_shadow.jpg'
img_file = 'signs_vehicles_xygrad.png'
img_file = 'curved-lane.jpg'
img_file = 'test6.jpg'

image = mpimg.imread('test_images/'+img_file)



def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    ksize = 15  # Choose a larger odd number to smooth gradient measurements
    img = np.copy(img)
    s_channel = color_select(img, 'S')
    h_channel = color_select(img, 'H')
    r_channel = color_select(img, 'R')

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize)
    dir_binary = dir_threshold(image, sobel_kernel=ksize)
    combined = np.zeros_like(s_channel)
    combined[((r_channel == 1) & (dir_binary == 1))] = 1
    line_img = hough_lines(combined*255,min_line_len=5, max_line_gap=1)
    prespective = get_source_dist_points_test(r_channel*255)
    S, D = get_source_dist_points(r_channel*255)
    unwarped_image = unwrap(r_channel, S, D)
    #hist, bin_edges = np.histogram(unwarped_image, bins=10)
    #hist = np.hstack(unwarped_image)
    #plt.hist(hist, bins='auto')
    #histr = cv2.calcHist([unwarped_image], [0], None, [256], [0, 256])
    sum = hist(unwarped_image)
    plt.plot(sum)
    #plt.hist(hist, bins='auto')
    #plt.hist(unwarped_image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')  # calculating histogram

    imgs =["s_channel", "h_channel", "r_channel", "gradx", "dir_binary", "combined", "line_img", "prespective", "unwarped_image"]
    img_dic = OrderedDict()
    for img in imgs:
        img_dic[img] = eval(img)
    draw_sub_plots(img_dic, col_no=3, raw_no=3, title=img_file)

    return stack_binary_images(line_img[:,:,0]*255, gradx, s_channel)


# Edit this function to create your own pipeline.
def pipeline_(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,ksize=15)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0,1, ksize=15)  # Take the derivative in y
    abs_sobely = np.absolute(sobely)  # Absolute y derivative to accentuate lines away from horizontal

    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_slope = np.arctan2(abs_sobely, abs_sobelx)
    binary_sobel_slope = np.zeros_like(sobel_slope)
    binary_sobel_slope[(sobel_slope >= .7) & (sobel_slope <= 1.3)] = 1
    kernel_size = 5
    blur_binary_sobel_slope = gaussian_blur(binary_sobel_slope, kernel_size)
    blur_binary_sobel_slope[blur_binary_sobel_slope <= 0] = 0
    #blur_binary_sobel_slope = cv2.Sobel(binary_sobel_slope, cv2.CV_64F, 1, 0,ksize=3)
    # Define our parameters for Canny and apply
    #low_threshold = 50
    #high_threshold = 100
    #edges = canny(np.uint8(blur_binary_sobel_slope), low_threshold, high_threshold)


    #Hough transform --> get the lines
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = (np.pi / 180)*1
    threshold = 75
    min_line_length = 10
    max_line_gap = 10
    line_img = hough_lines(np.uint8(blur_binary_sobel_slope), rho, theta, threshold, min_line_length, max_line_gap)
    color_line_img = np.dstack(((line_img[:, :, 0].astype(float) ), blur_binary_sobel_slope, np.zeros_like(blur_binary_sobel_slope)))



    imgs = [["binary_sobel_slope","blur_binary_sobel_slope"], ["line_img", "color_line_img"]]
    col_no = 2
    raw_no = 2
    f, ax = plt.subplots(raw_no, col_no, figsize=(24, 9))
    f.tight_layout()
    for c in range(0,col_no):
        for r in range(0, raw_no):
            #plt.imshow(exec(im), cmap='gray')
            img1_t =imgs[r][c]
            img1 = eval(img1_t)
            ax[r][c].imshow(img1, cmap='gray')
            ax[r][c].set_title(img1_t, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    print(np.shape(line_img[:,:,0]))
    print(np.shape(sxbinary))
    color_binary = np.dstack(((line_img[:,:,0].astype(float)/255), sxbinary, s_binary)) * 255
    return color_binary


def main():
    global image
    mtx, dist = load_calib()
    image = cv2.undistort(image, mtx, dist, None, mtx)

    result = pipeline(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    main()
    pass
