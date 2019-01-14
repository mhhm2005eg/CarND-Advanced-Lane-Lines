# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 00:05:04 2018

@author: mhhm2
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import _thread

from common import *
from common import hist, TEST_MODE, draw_sub_plots, curve_resynch_threashold
from calibration import load_calib
#from gradient import abs_sobel_thresh, mag_thresh, dir_threshold
#from color_space import color_select, stack_binary_images
from perspective_transformation import get_source_dist_points, get_source_dist_points_test, unwrap

from sobel import sobel
from color import color
from line import Line


class image(object):
    count = 0
    src_points = []
    dest_points = []
    M = []
    Minv = []
    img_size = ()
    curve_meter = 0
    poly_resynch = True
    max_cr = 0
    max_cl = 0
    img_max_cl =[]
    img_max_cr =[]
    img_final_max_cl = []
    img_final_max_cr = []
    count_max_cl = 0
    count_max_cr = 0
    store = {}
    brightness = []
    vertices = ()
    xsize = 0
    ysize = 0
    def __init__(self, caller=None, _image=None, image_path=None):
        self.image = _image
        self.caller = caller
        self.ksize = 15
        self.image_path = image_path
        if TEST_MODE[0] == "IMAGE":
            if image_path:
                self.file_name = os.path.splitext(os.path.basename(self.image_path))[0]
            if not (image_path is None):
                if os.path.isfile(image_path):
                    self.image = mpimg.imread(image_path)
                else:
                    print("could not load image: %s" %(image_path))
            else:
                print("could not load image: %s" %(image_path))
        self.ysize = self.image.shape[0]
        self.xsize = self.image.shape[1]
        if self.__class__.count == 0:
            self.__class__.src_points, self.__class__.dest_points = self.__class__.get_source_dist_points(self.image)
            self.__class__.getPerspectiveTransform()
            self.__class__.img_size = (self.image.shape[1], self.image.shape[0])
            self.__class__.xsize = self.__class__.img_size[0]
            self.__class__.ysize = self.__class__.img_size[1]
            self.__class__.get_vertices()
        self.__class__.count += 1

        self.i_image = np.copy(self.image)

        self.undistort()

        #colors
        self.red   = color(self, "R")
        self.green = color(self, "G")
        self.blue  = color(self, "B")
        self.saturation = color(self, "S")
        self.light = color(self, "L")
        self.hue = color(self, "H")
        self.gray = color(self, "Gr")

        self.masked_image = self.region_of_interest()
        self.combined = None

        #Gradient
        self.sobel = sobel(self)


    def pipeline(self):
        #ksize = 15  # Choose a larger odd number to smooth gradient measurements
        #img = np.copy(self.masked_image)
        #self.image = self.masked_image


        s_channel = self.saturation.get(binary=True, masked=True)
        if (not DEBUG) & (TEST_MODE[0] == "IMAGE"):
            h_channel = self.hue.get(binary=True, masked=True)
        l_channel = self.light.get(binary=True, masked=True)
        r_channel = self.red.get(binary=True, masked=True)

        gradx = self.sobel.abs_sobel_thresh( orient='x', sobel_kernel=self.ksize, masked=True)
        if (not DEBUG) & (TEST_MODE[0] == "IMAGE"):
            grady = self.sobel.abs_sobel_thresh( orient='y', sobel_kernel=self.ksize, masked=True)
            mag_binary = self.sobel.mag_thresh(sobel_kernel=self.ksize, masked=True)
        dir_binary = self.sobel.dir_threshold(sobel_kernel=self.ksize, masked=True)
        combined0 = np.zeros_like(s_channel)
        combined0[((r_channel == 1) & (s_channel == 1) & (dir_binary == 1))] = 1

        #store_image(combined0*255, self.file_name)

        combined1 = np.zeros_like(s_channel)
        combined1[((gradx == 1) & (dir_binary == 1))] = 1
        l2_channel = self.get_color(lower_range=[10,0,200], upper_range=[255,255,255], masked=True)
        #l2_channel_all = self.get_color(lower_range=[10,0,200], upper_range=[255,255,255], masked=False)

        combined2 =  np.zeros_like(s_channel)
        #print(np.max(dir_binary))
        combined2 = ( s_channel  + ((l_channel==0)& dir_binary.astype(bool)))*(255/2)
        combined2[combined2 < 100] = 0
        combined2[combined2 >= 100] = 1

        combined = image.image_or(combined1, combined2)

        yellow_white = self.get_yellow_and_white()


        #print(np.max(l2_channel), np.max(combined))
        l2_channel_binary = np.zeros_like(l2_channel)
        l2_channel_binary[(l2_channel > 0)] = 1

        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)

        l3_channel = region_of_interest(hsv[:,:,2],image.vertices)
        l3_channel_binary = np.zeros_like(l3_channel)
        l3_channel_binary[(l3_channel > 0)] = 1

        my_yellow = self.get_yellow()
        brightness = np.sum(l3_channel)/np.sum(l3_channel_binary)

        if TEST_MODE[0] == "IMAGE":
            print("brightness :" , brightness)


        image.brightness.append(brightness)

        if TEST_MODE[0] == "IMAGE":
            Mix = image.stack_binary_images(combined, yellow_white, l2_channel_binary)
            Mix2=cv2.addWeighted(self.image, .5, Mix, 1, 0)

            #self.combined = l2_channel_binary #image.image_and(combined, yellow_white)

        # if bright image
        if brightness > 150:
            self.combined = image.image_and(combined, yellow_white)
            self.combined = image.image_or(self.combined, gradx)

        elif brightness > 130:
            self.combined = image.image_or(s_channel, l2_channel_binary)

        # if normal
        elif brightness > 100:
            self.combined = s_channel

        elif brightness > 80:
            self.combined = l2_channel_binary
        #if dark image
        elif brightness < 80:
            self.combined = image.image_or(yellow_white, gradx)

        self.combined = image.image_or(self.combined, my_yellow)

        if (TEST_MODE[0] == "IMAGE"):
            line_img = hough_lines(self.combined * 255, min_line_len=5, max_line_gap=1)
        if (TEST_MODE[0] == "IMAGE"):
            prespective = get_source_dist_points_test(self.combined * 255)
        #S, D = get_source_dist_points(combined * 255)
        #unwarped_image = image.unwrap(line_img[:,:,0])
        unwarped_image = image.unwrap(self.combined)
        if 0:

            hist = np.hstack(unwarped_image)
            plt.hist(hist, bins='auto')

        # histr = cv2.calcHist([unwarped_image], [0], None, [256], [0, 256])
        if (DEBUG) & (TEST_MODE[0] == "IMAGE"):
            hist, bin_edges = np.histogram(unwarped_image, bins=10)
            fig1 = plt.figure(1)
            sum = hist(unwarped_image)
            plt.plot(sum)
        # plt.hist(hist, bins='auto')
        # plt.hist(unwarped_image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')  # calculating histogram
        # Get the exact Lane lines by the sliding windows
        Lanes_obj = Line(self, unwarped_image)
        lane_image = Lanes_obj._fit_polynomial()
        left_curverad, right_curverad = Lanes_obj.measure_curvature_meter()
        #print(left_curverad, right_curverad)
        curve_meter = np.mean([ left_curverad, right_curverad ])


        old_synch = self.__class__.poly_resynch
        if (TEST_MODE[0] == "VIDEO"):
            if (abs(curve_meter - self.__class__.curve_meter) > curve_resynch_threashold):
                self.__class__.poly_resynch = True
            else:
                self.__class__.poly_resynch = False
            self.__class__.curve_meter = curve_meter

            if (abs(left_curverad - right_curverad) > curve_resynch_threashold):
                self.__class__.poly_resynch = True


        lane_offset = Lanes_obj.get_offset()
        #
        #final = Lanes_obj.project("Radius of Curvature: L = %d (m)   R = %d (m) %d" %(left_curverad, right_curverad, old_synch), "Vehicle is %.2fm left of center" %lane_offset)
        final = Lanes_obj.project("Radius of Curvature %d (m) " %(curve_meter), "Vehicle is %.2fm left of center" %lane_offset)

        if not ("ll" in self.__class__.store):
            self.__class__.store["ll"] = []
        if not ("rr" in self.__class__.store):
            self.__class__.store["rr"] = []
        self.__class__.store["ll"].append(left_curverad)
        self.__class__.store["rr"].append(right_curverad)

        if (TEST_MODE[0] == "VIDEO"):
            original_image = self.i_image
            undistorted_image = self.image

            imgs = ["original_image", "undistorted_image", "final", "lane_image", "s_channel", "r_channel", "gradx", "combined2",
                    "dir_binary", "combined", "combined1",
                    "unwarped_image", "yellow_white"]
            img_dic = OrderedDict()
            for img in imgs:
                img_dic[img] = eval(img)


            if left_curverad > self.__class__.max_cl:
                self.__class__.img_max_cl = np.copy(self.image)
                self.__class__.img_final_max_cl = np.copy(final)
                self.__class__.max_cl = left_curverad
                self.__class__.count_max_cl = self.__class__.count
                self.__class__.store["left"] = img_dic.copy()

            if right_curverad > self.__class__.max_cr:
                self.__class__.img_max_cr = np.copy(self.image)
                self.__class__.img_final_max_cr = np.copy(final)
                self.__class__.max_cr = right_curverad
                self.__class__.count_max_cr = self.__class__.count
                self.__class__.store["right"] = img_dic.copy()



        if TEST_MODE[0] == "IMAGE":
            original_image = self.i_image
            undistorted_image = self.image

            imgs = ["original_image", "undistorted_image","final","lane_image","s_channel", "h_channel", "r_channel", "gradx", "combined2","dir_binary", "combined","combined1" ,"line_img",
                    "unwarped_image", "h_channel", "prespective", "yellow_white", "self.combined", "l2_channel", "l_channel", "Mix", "Mix2", "l3_channel", "my_yellow"]
            img_dic = OrderedDict()
            for img in imgs:
                img_dic[img] = eval(img)
            #draw_sub_plots(img_dic, col_no=4, raw_no=4, title=self.file_name)
            store_images(img_dic, title=self.file_name)
            write_json_file({"brightness":brightness}, self.file_name)
        if TEST_MODE[0] == "IMAGE":
            store_image(final, self.file_name)

        return final #stack_binary_images(line_img[:, :, 0] * 255, gradx, s_channel)

    def undistort(self):
        #print("undistort", self.caller.mtx, self.caller.dist)
        self.image = cv2.undistort(self.image, self.caller.mtx, self.caller.dist, None, self.caller.mtx)

    @staticmethod
    def get_source_dist_points(img, n=PRESPECTIVE_N):
        #img = np.copy(img)
        ysize = img.shape[0]
        xsize = img.shape[1]
        # Mask the edges

        cut = 20

        if 0:
            s1 = (3.5 * int(xsize / cut), int(ysize))
            s2 = (int(xsize / 2 - 2 * xsize / (2 * cut)), int(ysize / 2 + 3 * ysize / cut))
            s3 = (int(xsize / 2 + 1.25 * xsize / cut), int(ysize / 2 + 3 * ysize / cut))
            s4 = (int(xsize - 2.75 * xsize / (cut)), int(ysize))
            dx1 = 6 * int(xsize / cut)
            dx2 = int(xsize - 5 * xsize / (cut))
            d1 = (dx1, int(ysize))
            d2 = (dx1, int(0))
            d3 = (dx2, int(0))
            d4 = (dx2, int(ysize))

        else:
            s1 = (278 - n, 678)
            s2 = (601 - n, 446 + n)
            s3 = (680 + n, 446 + n)
            s4 = (1035 + n, 678)

            dx1 = 6 * int(xsize / cut) - n
            dx2 = int(xsize - 6 * xsize / (cut)) + n
            d1 = (dx1, int(ysize))
            d2 = (dx1, int(n))
            d3 = (dx2, int(n))
            d4 = (dx2, int(ysize))

        return np.float32([s1, s2, s3, s4]), np.float32([d1, d2, d3, d4])

    @staticmethod
    def getPerspectiveTransform():
        image.M = cv2.getPerspectiveTransform(image.src_points, image.dest_points)
        image.Minv = cv2.getPerspectiveTransform(image.dest_points, image.src_points)

    @staticmethod
    def unwrap(gray, src, dst):
        img_size = (gray.shape[1], gray.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)
        # Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    @staticmethod
    def unwrap(gray):
        unwarped = cv2.warpPerspective(gray, image.M, image.img_size, flags=cv2.INTER_LINEAR)
        return unwarped

    @staticmethod
    def wrap(gray):
        warped = cv2.warpPerspective(gray, image.Minv, image.img_size, flags=cv2.INTER_LINEAR)
        return warped

    def region_of_interest(self, _image=None, vertices = None, cut = 20, wide = 1):
        if _image is None:
            _image = self.image
        #Mask the edges
        if vertices is None:
            vertices = image.vertices

        masked_image = region_of_interest(_image, vertices)

        return masked_image

    @staticmethod
    def get_vertices(cut = 20, wide = 1):
        p1 = (int(2 * image.xsize / cut), int(image.ysize - image.ysize / 10))
        p2 = (int(image.xsize / 2 - (wide) * image.xsize / cut), int(image.ysize / 2 + 2 * image.ysize / cut))
        p3 = (int(image.xsize / 2 + (wide) * image.xsize / cut), int(image.ysize / 2 + 2 * image.ysize / cut))
        p4 = (int(image.xsize - 2 * image.xsize / cut), int(image.ysize - image.ysize / 10))
        vertices = np.array([[p1, p2, p3, p4]], dtype=np.int32)
        image.vertices = vertices
        return vertices

    @staticmethod
    def filter_RGB(image, r_th=(0,255), g_th=(0,255), b_th=(0,255)):
        img = np.copy(image)
        r = 0
        g = 1
        b = 2
        outed = np.where(((image[:,:,r] > r_th[1])|(image[:,:,r] < r_th[0])) | ((image[:,:,g] > g_th[1])|(image[:,:,g] < g_th[0])) | ((image[:,:,b] > b_th[1])|(image[:,:,b] < b_th[0])))

        img[outed] = 0

        return img

    @staticmethod
    def convert_figure_to_array(fig0):
        fig0.canvas.draw()
        data = np.fromstring(fig0.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        w, h = fig0.canvas.get_width_height()
        result = data.reshape((h, w, 3))
        return result

    @staticmethod
    def clear ():
        image.count_max_cl = 0
        image.max_cl = 0
        image.count_max_cr = 0
        image.max_cr = 0
        image.store = {}
        image.brightness = []


    def get_yellow_and_white(self, masked=True):
        image = self.image
        #lower_range_yellow_HSV = np.array([20, 100, 100], dtype=np.uint8)
        #upper_range_yellow_HSV = np.array([40, 255, 255], dtype=np.uint8)
        #lower_white = np.array([0, 0, 210])
        #upper_white = np.array([180, 25, 255])

        lower_range_yellow_HSV = np.array([20, 50, 100], dtype=np.uint8)
        upper_range_yellow_HSV = np.array([40, 255, 255], dtype=np.uint8)
        lower_white = np.array([0,20, 210])
        upper_white = np.array([180, 255, 255])

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask_yellow_hsv = cv2.inRange(hsv, lower_range_yellow_HSV, upper_range_yellow_HSV)
        mask_white_hsv = cv2.inRange(hsv, lower_white, upper_white)

        combined = np.zeros_like(mask_yellow_hsv)
        combined[(mask_yellow_hsv >= 1) | (mask_white_hsv >= 1)] = 1

        if DEBUG:
            # plt.imshow(hue, 'gray')
            imgs = ["image", "combined", "mask_yellow_hsv", "mask_white_hsv"]
            img_dic = OrderedDict()
            for img in imgs:
                img_dic[img] = eval(img)
            draw_sub_plots(img_dic, col_no=2, raw_no=2, title="color test")
            plt.show()

        if masked:
            combined = self.region_of_interest(combined)


        return combined

    def get_yellow(self, ower_range=[20, 50, 100], upper_range=[40, 255, 255] ,masked=True):
        return self.get_color(ower_range, upper_range, masked)

    def get_color(self,lower_range=[20, 50, 100], upper_range=[40, 255, 255] ,masked=True):
        image = self.image
        #lower_range_yellow_HSV = np.array([20, 100, 100], dtype=np.uint8)
        #upper_range_yellow_HSV = np.array([40, 255, 255], dtype=np.uint8)
        #lower_white = np.array([0, 0, 210])
        #upper_white = np.array([180, 25, 255])

        lower_range = np.array(lower_range, dtype=np.uint8)
        upper_range = np.array(upper_range, dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        masked_hsv = cv2.inRange(hsv, lower_range, upper_range)

        if masked:
            combined = self.region_of_interest(masked_hsv)
        else:
            combined = masked_hsv


        return combined

    @staticmethod
    def image_or(img1, img2):
        ret = np.zeros_like(img1)
        ret[(img1 >= 1 )| (img2 >= 1 )] = 1
        return ret

    @staticmethod
    def image_and(img1, img2):
        ret = np.zeros_like(img1)
        ret[(img1 >= 1 ) & (img2 >= 1 )] = 1
        return ret

    @staticmethod
    def stack_binary_images(R=[], G=[], B=[], factor=255):

        ref = None

        if R.any():
            ref = R
        elif G.any():
            ref = G
        elif B.any():
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
