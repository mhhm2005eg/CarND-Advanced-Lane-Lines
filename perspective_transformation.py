import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import *
from color_space import color_select, draw_sub_plots, stack_binary_images



# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found:
    # a) draw corners
    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    # Note: you could pick any four of the detected corners
    # as long as those four corners define a rectangle
    # One especially smart way to do this would be to use four well-chosen
    # corners that were automatically detected during the undistortion steps
    # We recommend using the automatic detection of corners in your code
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    # img = cv2.imread(file_path)

    img = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 2), np.float32)
    objp[:, :2] = np.mgrid[1:nx+1, 1:ny+1].T.reshape(-1, 2)

    img_size = (img.shape[1], img.shape[0])
    Coxeter = img_size[0]/(nx+1)
    dest = np.float32([objp[0], objp[7], objp[40], objp[47]])
    offset = np.float32([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
    #print(dest)

    dest = dest + offset
    #print(dest)
    #src = np.float32([corners[0], corners[7], corners[40], corners[47]])
    src = np.float32([corners[0], corners[nx - 1], corners[-nx], corners[-1]])
    dst1 = np.float32(dest)*Coxeter
    offset = 100
    #dst1 = np.float32([[offset, offset], [img_size[0] - offset, offset], [offset, img_size[1] - offset], [img_size[0] - offset, img_size[1] - offset]])
    print(len(src), len(dst1))
    if False:
        plt.imshow(img)

        plt.plot(*dst1[0],'*')
        plt.plot(*dst1[1],'*')
        plt.plot(*dst1[2],'*')
        plt.plot(*dst1[3],'*')
    M = cv2.getPerspectiveTransform(src, dst1)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M

# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp1(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    plt.imshow(undist)
    plt.show()
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M
def main():
    top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)

    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def get_source_dist_points_test(img):
    img = np.copy(img)
    img1 = np.copy(img)
    (s1, s2, s3, s4), (d1, d2, d3, d4) = get_source_dist_points(img)
    vertices = np.array([[s1,  s2,  s3, s4]], dtype=np.int32)
    #masked_edges = region_of_interest(img, vertices, (100,100,100))
    lines_s = [[list(np.concatenate([s1, s2])), list(np.concatenate([s2 , s3])), list(np.concatenate([s3 , s4])), list(np.concatenate([s1 , s4]))]]
    lines_d = [[list(np.concatenate([d1 , d2])), list(np.concatenate([d2 , d3])), list(np.concatenate([d3 , d4])), list(np.concatenate([d1 , d4]))]]
    lines = lines_s + lines_d
    draw_lines(img, lines, color=[255, 255, 255], thickness=5)

    img = stack_binary_images(np.array([]),img1*255, img*255 )
    return img

def get_source_dist_points(img, n=PRESPECTIVE_N):
    img = np.copy(img)
    ysize = img.shape[0]
    xsize = img.shape[1]
    #Mask the edges
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
        dx2 = int(xsize - 5 * xsize / (cut)) + n
        d1 = (dx1, int(ysize))
        d2 = (dx1, int(n))
        d3 = (dx2, int(n))
        d4 = (dx2, int(ysize))

    return np.float32([s1, s2, s3, s4]), np.float32([d1, d2, d3, d4])


def unwrap(gray, src, dst):
    img_size = (gray.shape[1], gray.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("calibration_wide/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    img = cv2.imread('calibration_wide/GOPR0032.jpg')
    nx = 8  # the number of inside corners in x
    ny = 6  # the number of inside corners in y

    main()
    pass
