# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:49:07 2018

@author: mhhm2
"""
import os,sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from common import *
from common import hist
from calibration import load_calib
from gradient import abs_sobel_thresh, mag_thresh, dir_threshold
from color_space import color_select, draw_sub_plots, stack_binary_images
from perspective_transformation import get_source_dist_points, get_source_dist_points_test, unwrap
DEBUG = 0
#TEST = "VIDEO_TEST"
from test_images import image as image
class video(object):

    def __init__(self, caller, vid_path=None):
        #super().__init__()
        self.caller = caller
        self.mtx =  self.caller.mtx
        self.dist =  self.caller.dist
        if vid_path:
            self.vid_path = vid_path
            self.vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        else:
            self.vid_path = None
            self.vid_name = None
    def image_wrapper(self, _image):
        img = image(self, _image=_image)
        ret = img.pipeline()

        return ret


    def test_video(self, vid_path=None):
        if type(vid_path) == list:
            for path in vid_path:
                self.test_video(path)
            return
        if not vid_path:
            vid_path = self.vid_path
        print(type(vid_path))
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        white_output = main_dir+"/"+vid_out_dir+"/"+vid_name+"_output.mp4"
        #white_output = os.path.splitext(os.path.basename(vid_path))[0]+"_ouput.mp4"
        print(vid_path)
        clip1 = VideoFileClip(vid_path)
        video_clip = clip1.fl_image(self.image_wrapper)
        video_clip.write_videofile(white_output, audio=False)
        video_clip.reader.close()

        store_images(image.store["left"], "left", MainDir=vid_out_dir+"/"+ vid_name )
        store_images(image.store["right"], "right", MainDir=vid_out_dir+"/"+ vid_name )
        fig0 = plt.figure(0)
        plt.plot(image.store["ll"], label='left_curves')
        plt.plot(image.store["rr"], label='right_curves')
        plt.grid(True)
        plt.legend()
        arr = image.convert_figure_to_array(fig0)
        store_image(arr, "curve",vid_out_dir + "/" + vid_name)
        fig0.clf()

        fig0 = plt.figure(0)
        fig0.clf()
        #print(image.brightness)
        plt.plot(image.brightness, label='brightness')
        plt.grid(True)
        plt.legend()
        arr = image.convert_figure_to_array(fig0)
        store_image(arr, "brightness",vid_out_dir + "/" + vid_name)
        print("Avg brightness: " + str(np.average(image.brightness)))
        min_ = np.min(image.brightness)
        max_ = np.max(image.brightness)
        print("Min brightness: " + str(min_), str(image.brightness.index(min_)))
        print("MAX brightness: " + str(max_), str(image.brightness.index(max_)))
        fig0.clf()

        print(image.count_max_cl,":" ,image.max_cl,image.count_max_cr, ":", image.max_cr)
        image.clear()
        video_clip.audio.reader.close_proc()




