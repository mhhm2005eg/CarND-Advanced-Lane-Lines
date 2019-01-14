import os
from importlib import reload  # Python 3.4+ only.

#-----------------------------------------------------
main_dir = os.path.dirname(os.path.abspath(__file__))
lane_width = 3.7
dashed_line_length = 3
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
#-----------------------------------------------------


DEBUG = 0
img_form = "jpg"
img_out_dir = "./output_images"
vid_form = "mp4"
vid_out_dir = "./test_videos_output"
#-----------------------------------------------------

#TEST_MODE = ("VIDEO", "S")
TEST_MODE = ("IMAGE", "A")
#-----------------------------------------------------

USE_PROFILER = False
USE_THREADING = False

SHORT_SCAN = False
#from conf import load_config
#load_config("challenge_video")
#from conf import SHORT_SCAN

#-----------------------------------------------------

curve_resynch_threashold = 300

#-----------------------------------------------------
if SHORT_SCAN:
    PRESPECTIVE_N = 50
    LANE_PERCENTAGE_SCANE = 0.6
else:
    PRESPECTIVE_N = 0
    LANE_PERCENTAGE_SCANE = 0.5

#-----------------------------------------------------

