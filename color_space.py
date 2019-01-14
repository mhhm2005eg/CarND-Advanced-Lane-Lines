import numpy as np
import cv2
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from inliner import inline

from common import *


img_file = 'test_images/test6.jpg'
#image = mpimg.imread(img_file)
threshold = {'R':(200, 255), 'G':(200, 255), 'B':(200, 255), 'H':(15, 100), 'L':(0,255), 'S':(90, 255)}
def test():
    thresh = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_binary = np.zeros_like(gray)
    gray_binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    thresh = (200, 255)
    red_binary = np.zeros_like(R)
    red_binary[(R > thresh[0]) & (R <= thresh[1])] = 1

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    thresh = (90, 255)
    S_binary = np.zeros_like(S)
    S_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    thresh = (15, 100)
    H_binary = np.zeros_like(H)
    H_binary[(H > thresh[0]) & (H <= thresh[1])] = 1

    imgs = ["gray", "gray_binary", "R", "red_binary", "S", "S_binary", "H", "H_binary"]
    img_dic = OrderedDict()
    col_no = 4
    raw_no = 2

    #imgs = [[gray, gray_binary, R, red_binary ], [S, S_binary, H, H_binary]]
    #imgs = [[gray, gray_binary, R, red_binary ], [S, S_binary, H, H_binary]]
    for img in imgs:
        img_dic[img] = eval(img)
    draw_sub_plots(img_dic, col_no=4, raw_no=2)
if 0:
    imgs = ["gray", "gray_binary", "R", "red_binary", "S", "S_binary", "H", "H_binary"]

    imgs += [""]*((col_no*raw_no) -len(imgs))
    imgs = np.reshape(imgs, (raw_no, col_no))


    f, ax = plt.subplots(raw_no, col_no)
    #f.tight_layout()
    for c in range(0,col_no):
        for r in range(0, raw_no):
            #plt.imshow(exec(im), cmap='gray')
            if imgs[r][c]:
                ax[r][c].imshow(eval(imgs[r][c]), cmap='gray')
                ax[r][c].set_title(imgs[r][c], fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binary_output


def color_select(img, color='R', binary = True, thresh=None):
    #image received is RGB  mpimg.imread
    RGB_colors = {'R':0, 'G':1, 'B':2}
    HLS_colors = {'H':0, 'L':1, 'S':2}
    if color in RGB_colors:
        channel = img[:,:,RGB_colors[color]]
    elif color in HLS_colors:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        channel = img[:, :, HLS_colors[color]]
    if binary:
        if not thresh:
            thresh = threshold[color]

        binary_output = np.zeros_like(img[:,:,0])
        binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        return binary_output
    else:
        return channel


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
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    if not (title is None):
        f.suptitle(title, fontsize=20)

    plt.show()


def stack_binary_images(R=[], G=[], B=[], factor=255):

    ref = None

    if R.any() :
        ref = R
    elif G.any() :
        ref = G
    elif B.any() :
        ref = B
    else:
        pass

    if not R.any():
        R = np.zeros_like(ref)

    if not G.any():
        G = np.zeros_like(ref)

    if not B.any():
        B = np.zeros_like(ref)


    color_binary = np.dstack((R, G, B)) * factor

    return color_binary



def main():
    pass
    test()


# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    main()
    pass
