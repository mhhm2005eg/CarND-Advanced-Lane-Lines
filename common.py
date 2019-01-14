#importing some useful packages
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline
import math
from collections import OrderedDict
from PIL import Image
import glob
import json

from configuration import *

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def save_image(name, form="jpg"):
    name = img_out_dir + "/" + name + "." + form
    print ("Saving image: " + name)
    plt.savefig(name, format=form)
    plt.close()
    return name


def store_image(image, name, dir=None):
    if not dir:
        dir = img_out_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    name = dir + "/" + name + "." + img_form
    print("Saving image: " + name)
    if np.max(image) == 1:
        image = image * 255

    im = Image.fromarray(image)
    if im.mode != 'RGB':
       im = im.convert('RGB')
    im.save(name)

def store_images(img_dic, title=None, MainDir=None):
    for key, _image in img_dic.items():
        if not MainDir:
            MainDir = img_out_dir
        directory = MainDir + "/" + title + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        store_image(_image, key , dir=directory)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, background=0):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    #if background != (0,0,0):
        #np.where(im == 230, 255, 0)
    if np.max(img) == 1:
        n = 1
    else:
        n =255
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (n,) * channel_count
    else:
        ignore_mask_color = n
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines = [], color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if not ( lines is None) :
        for line in lines:
            #print(lines, line)
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        print ("No lines detected ! Your HOUGH.")


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, slop_min = 0.5, slope_max = 100):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)


    lines = filter_lines(lines, slop_min, slope_max)
    lines = group_lines(lines, img.shape[0])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def hough_lines(img, rho=1, theta=(np.pi / 180)*1, threshold=30, min_line_len=30, max_line_gap=10):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    img = img.astype('uint8')
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    #slop_min = 0.5
    #slope_max = 100
    #lines = filter_lines(lines, slop_min, slope_max)
    #lines = group_lines(lines, img.shape[0])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



def conver_2_RGB(img, color=0):
    RGB_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    RGB_img[:,:,color] = img
    return RGB_img

def find_files(p=None, ext=".log", recursive=False):
    """
    Function to get all files of certain extensions within certain directory.
    :param p: path to scan
    :param ext: extension to search for.
    :return: list of files found.
    """

    ret = []
    if recursive:
        for root, dirs, files in os.walk(p):
            for loc_file in files:
                if loc_file.endswith(ext):
                    f = os.path.abspath(os.path.join(root, loc_file))
                    if os.path.isfile(f):
                        ret.append(f)
    else:
        ret = glob.glob(p+"/*"+ext)
    return ret

def save_image(name, img_form=img_form):
    """

    :param name:
    :return:
    """

    name = img_out_dir + "/" + name + "." + img_form
    print ("Saving image: " + name)
    plt.savefig(name, format=img_form)
    #plt.close()
    return name


def filter_lines(lines, slop_min = 0, slope_max = 200):
    ret = []
    for line in lines:
        #print(lines, line)
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            absslop = math.fabs(slope)
            if((absslop >= slop_min) and (absslop <=slope_max)):
                ret.append([[x1,y1,x2,y2]])
  
    #print (lines) 
    #print (ret)          
    return ret


def group_lines(lines, imshap0=900):
    ret = [[]]
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    min_y = int(imshap0 * (3 / 5))
    max_y = int(imshap0)
    
    for line in lines:
        #print(lines, line)
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
  
    if left_line_x:         
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        ret[0].append([left_x_start, max_y, left_x_end, min_y])
 
    if right_line_x:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        ret[0].append([right_x_start, max_y, right_x_end, min_y])
    
       
    return ret


def namestr(obj):
    x =  [name for name in globals() if globals()[name] is obj]
    return  x[0]


def draw_points(points):
    for p in points:
        plt.plot(*p, '*')


def measure_curvature_pixels(left_fit, right_fit, y):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    A = left_fit[0]
    B = left_fit[1]
    left_curverad = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / abs(2 * A)  ## Implement the calculation of the left line here

    A = right_fit[0]
    B = right_fit[1]
    right_curverad = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / abs(2 * A)  ## Implement the calculation of the right line here
    return left_curverad, right_curverad


def measure_curvature_meter(ploty, leftx, rightx):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    left_fit = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    A = left_fit[0]
    B = left_fit[1]

    y = np.max(ploty*ym_per_pix)
    left_curverad = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / abs(2 * A)  ## Implement the calculation of the left line here

    A = right_fit[0]
    B = right_fit[1]
    right_curverad = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / abs(2 * A)  ## Implement the calculation of the right line here

    return left_curverad, right_curverad

def draw_sub_plots(imgs, col_no = 4, raw_no = 2 , title = None):
    f, ax = plt.subplots(raw_no, col_no)
    vals = list(imgs.values())
    kys = list(imgs.keys())
    if col_no == 1:
        item = "ax[r]"
    elif raw_no ==1:
        item = "ax[c]"
    else:
        item = "ax[r][c]"

    for r in range(0, raw_no):
        for c in range(0,col_no):
            index = r*col_no+c
            if (index < len(kys)):
                eval(item).imshow( vals[index], cmap='gray')
                eval(item).set_title(kys[index], fontsize=20)
    #f.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    if not (title is None):
        f.suptitle(title, fontsize=20)

    save_image("group_" + title )
    #plt.show()

def write_json_file(data, file_name, MainDir = None):
    """
    :param file_name: the file name to save
    :param data: data to store
    write data in a json file.
    :return:
    """
    # Common.print_system_info()
    # Get Configuration file path
    # ---------------------------,
    if not MainDir:
        MainDir = img_out_dir

    file_name = MainDir + "/" + file_name + "/"+ file_name
    ext = ".json"
    file_name += ext
    with open(file_name, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)

def rmse(ref_list, tested_list):
    ret = np.sqrt(
        ((np.array(ref_list) - np.array(tested_list)) ** 2).mean())
    return ret

def write_txt_file(data, file_name):
    """
    :param file_name: the file name to save
    :param data: data to store
    write data in a txt file.
    :return:
    """
    # Common.print_system_info()
    # Get Configuration file path
    # ---------------------------
    ext = ".txt"
    file_name += ext
    print ("Writing file: " + file_name)
    with open(file_name, 'w') as fp:
        t = str(data)
        fp.write(t)

