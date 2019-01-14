import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import _thread
import threading
import cProfile, pstats, io
#from pstats import SortKey

from common import *
from test_images import image
from test_videos import video
from calibration import load_calib


class master(object):
    def __init__(self):
        self.mtx = 0
        self.dist = 0
        self.image = object()
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)

        if not os.path.exists(vid_out_dir):
            os.makedirs(vid_out_dir)

    def test_images(self, pipe="pipeline"):
        images = find_files(main_dir + "/test_images/", "*.jpg")

        def _threading(img_path, pipe):
            img = image(self,  image_path=img_path)
            eval("img." + pipe)()

        for img_path in images:
            if USE_THREADING:
                try:
                    t = threading.Thread(target=_threading, args=(img_path, pipe, ))
                    t.start()
                except:
                    print("Error: unable to start thread")
            else:
                img = image(self, image_path=img_path)
                eval("img." + pipe)()
        fig0 = plt.figure(0)
        fig0.clf()
        print(image.brightness)
        plt.plot(image.brightness, label='brightness')
        plt.grid(True)
        plt.legend()
        arr = image.convert_figure_to_array(fig0)
        store_image(arr, "brightness",img_out_dir + "/" + "brightness")
        print("Avg brightness: " + str(np.average(image.brightness)))
        min_ = np.min(image.brightness)
        max_ = np.max(image.brightness)
        print("Min brightness: " + str(min_), images[image.brightness.index(min_)])
        print("MAX brightness: " + str(max_), images[image.brightness.index(max_)])
        fig0.clf()

    def test_videos(self):
        videos = find_files(main_dir + "/test_videos", "*.mp4")
        print(videos)
        for vid_path in videos:
            vid = video(self, vid_path)
            vid.test_video()


    def main(self):
        self.mtx, self.dist = load_calib("camera_cal")

        if TEST_MODE[0] == "IMAGE":
            if TEST_MODE[1] == "S":
                self.image = image(self, image_path="./test_images/test4.jpg")
                self.image.pipeline()
            else:
                self.test_images(pipe="pipeline")
        elif TEST_MODE[0] == "VIDEO":
            if TEST_MODE[1] == "S":
                vid = video(self)
                vid.test_video("./test_videos/project_video.mp4")
                #vid.test_video(["./test_videos/challenge_video.mp4", "./test_videos/harder_challenge_video.mp4"])

            else:
                self.test_videos()

    
if __name__ == "__main__":
    #lane_color()
    #lane_region()
    #lane_region_color()
    #canny_test1()
    Master = master()
    if not USE_PROFILER:
        Master.main()
    else:
        pr = cProfile.Profile()
        pr.enable()
        Master.main()
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open("profiler_Output.txt", "w") as text_file:
            text_file.write(s.getvalue())
        #cProfile.run('Master.main()',"profiler_out.txt")


