import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import *
# prepare object points
nx = 9
ny = 6

calib_folder_name = "calibration_wide"

def save_calib(mtx, dist, calib_folder_name=calib_folder_name):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(calib_folder_name+"/"+calib_folder_name+".p", "wb"))


def loc_load_calib_file(file_path):
    # Read in the saved objpoints and imgpoints
    print("loading file: " + file_path)
    dist_pickle = {}
    dist_pickle["mtx"] = []
    dist_pickle["dist"] = []
    dist_pickle = pickle.load(open(file_path, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist


def load_calib_file(folder_name):
    # Read in the saved objpoints and imgpoints
    file_path = folder_name+"/"+folder_name+".p"
    if os.path.isfile(file_path):
         return loc_load_calib_file(file_path)
    return np.array([None]), np.array([None])


def Calib_img(fname):
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if DEBUG:
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            plt.show()
    return ret, corners


def Calib_imgs(samples_folder=calib_folder_name):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    print("Configured nx = %d and ny = %d" % ( nx, ny))
    imgs = find_files("./" + samples_folder, "jpg")
    for img in imgs:
        print("processing image: " + img)
        ret, corners = Calib_img(img)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    t = cv2.imread(img)
    img_size = (t.shape[1], t.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    save_calib(mtx, dist, samples_folder)

    return mtx, dist


def load_calib(samples_folder=calib_folder_name):
    mtx, dist = load_calib_file(samples_folder)
    if  mtx.all() == None:
        mtx, dist = Calib_imgs(samples_folder)
    return mtx, dist


def undistort_image(file_path, mtx, dist):
    img = cv2.imread(file_path)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def undistort_images(test_samples_folder):
    mtx, dist = load_calib()
    imgs = find_files("./" + test_samples_folder, "jpg")
    for img in imgs:
        dst = undistort_image(img, mtx, dist)
        des_img = img.replace(test_samples_folder, test_samples_folder+"_out")
        print("Writing image: " + des_img)
        cv2.imwrite(des_img, dst)


def wrap_images(test_samples_folder):
    mtx, dist = load_calib()
    imgs = find_files("./" + test_samples_folder, "jpg")
    for img in imgs:
        dst = undistort_image(img, mtx, dist)
        des_img = img.replace(test_samples_folder, test_samples_folder+"_out")
        print("Writing image: " + des_img)
        cv2.imwrite(des_img, dst)

def main():
    undistort_images("test_images")


# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    main()
    pass
