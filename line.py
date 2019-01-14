import numpy as np
import cv2
import matplotlib
#matplotlib.use('Agg') # None interactive backend
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from common import DEBUG, draw_points, draw_sub_plots, hough_lines, USE_THREADING, TEST_MODE, LANE_PERCENTAGE_SCANE
from collections import OrderedDict


# Define a class to receive the characteristics of each line detection
class Line(object):
    lane_width = 3.7
    dashed_line_length = 3
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = (30 / 720)  # meters per pixel in y dimension
    xm_per_pix = (3.7 / 700)  # meters per pixel in x dimension
    lane_width_pix = 700
    count = 0
    ploty = 0
    left_fit = []
    right_fit = []

    def __init__(self, caller, binary_warped=None, image=None,  nwindows = 9, margin = 75, minpix = 300):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.binary_warped = binary_warped
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.leftx_delta = 0
        self.lefty_delta = 0
        self.rightx_delta = 0
        self.righty_delta = 0
        self.left_fitx = np.array([], dtype=int)
        self.right_fitx = np.array([], dtype=int)
        self.leftx = []
        self.rightx = []
        self.caller = caller
        self.image = self.caller.image
        self.left_curverad = 0
        self.right_curverad = 0
        self.window_height = 0
        if self.__class__.count == 0:
            self.__class__.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        self.__class__.count += 1

    def _find_lane_pixels(self, binary_warped, base=LANE_PERCENTAGE_SCANE): # base could be o.5
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]*base):, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // (2))

        if histogram[:midpoint].any():
            leftx_base = np.argmax(histogram[:midpoint])
            left_found = True
        else:
            left_found = False


        if histogram[midpoint:].any():
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            right_found = True
        else:
            right_found = False



        if (left_found)&(not right_found):
            rightx_base = leftx_base + Line.lane_width_pix
        elif (not left_found) & ( right_found):
            leftx_base = rightx_base - Line.lane_width_pix
        elif  (not left_found) & (not right_found):
            leftx_base = np.int(histogram.shape[0] // (4))
            rightx_base = np.int(3*histogram.shape[0] // (4))


            #print(leftx_base, rightx_base, binary_warped.shape[1] //2)
        left_lane_indsx = list()
        left_lane_indsy = list()
        right_lane_indsx = list()
        right_lane_indsy = list()

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = self.nwindows
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // (nwindows))
        self.window_height = window_height
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        lefty_current = binary_warped.shape[0] - 1
        righty_current = binary_warped.shape[0] - 1
        if 0:
            left_lane_indsx.append(leftx_current)
            left_lane_indsy.append(lefty_current)
            right_lane_indsx.append(rightx_current)
            right_lane_indsy.append(righty_current)
            plt.plot(leftx_current, lefty_current, '*')
            plt.plot(rightx_current, righty_current, '*')
        X_MAX = self.caller.__class__.img_size[0] - 1
        Y_MAX = self.caller.__class__.img_size[1] - 1
        #print(X_MAX)
        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - int(window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = int(leftx_current - margin)  # Update this
            win_xleft_high = int(leftx_current + margin)  # Update this
            win_xright_low = int(rightx_current - margin)  # Update this
            win_xright_high = int(rightx_current + margin)  # Update this

            if 0:
                if (win_xleft_high > X_MAX):
                    win_xleft_high = X_MAX

                if (win_xright_high > X_MAX):
                    win_xright_high = X_MAX

            if  not DEBUG:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            # plt.imshow(out_img)
            # plt.show()
            window_left_data = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
            window_right_data = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high]
            # plt.imshow(window_left_data, 'gray')
            # plt.show()

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = window_left_data.nonzero()
            good_right_inds = window_right_data.nonzero()

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # print(len(good_left_inds[0]), len(good_right_inds[0]))
            if len(good_left_inds[0]) >= minpix:
                histogram_left_x = np.sum(window_left_data, axis=0)
                histogram_left_y = np.sum(window_left_data, axis=1)
                self.leftx_delta = np.argmax(histogram_left_x)
                self.lefty_delta = np.argmax(histogram_left_y)
                leftx_current += self.leftx_delta + win_xleft_low - leftx_current
                lefty_current += self.lefty_delta + win_y_low - lefty_current
                left_x_OK = True
            else:
                left_x_OK = False
                pass
                #leftx_current += self.leftx_delta
                #lefty_current += self.lefty_delta

            if len(good_right_inds[0]) >= minpix:
                histogram_right_x = np.sum(window_right_data, axis=0)
                histogram_right_y = np.sum(window_right_data, axis=1)
                self.rightx_delta = np.argmax(histogram_right_x)
                self.righty_delta = np.argmax(histogram_right_y)
                rightx_current += self.rightx_delta + win_xright_low - rightx_current
                righty_current += self.righty_delta + win_y_low - righty_current
                right_x_OK = True
            else:
                pass
                right_x_OK = False
                #rightx_current += self.rightx_delta
                #righty_current += self.righty_delta

            if left_x_OK & (not right_x_OK):
                rightx_current += self.leftx_delta - margin
                righty_current += -self.lefty_delta - self.window_height
                self.rightx_delta = self.leftx_delta
                self.righty_delta = self.lefty_delta
                #print(-1)
            elif  (not left_x_OK) & ( right_x_OK):
                leftx_current += self.rightx_delta  - margin
                lefty_current += -self.righty_delta  - self.window_height
                self.leftx_delta = self.rightx_delta
                self.lefty_delta = self.righty_delta

                #print(1)
            elif (not left_x_OK) & (not right_x_OK):
                pass
                rightx_current += self.rightx_delta  - margin
                righty_current += -self.righty_delta - self.window_height
                leftx_current += self.leftx_delta  - margin
                lefty_current += -self.lefty_delta- self.window_height
                #print(0)
            else:
                pass
                #print(2)

            #righty_current = min(righty_current, binary_warped.shape[0]-1)
            #lefty_current  = min(lefty_current, binary_warped.shape[0]-1)

            if 0:
                if (leftx_current > X_MAX):
                    leftx_current = X_MAX

                if (rightx_current > X_MAX):
                    rightx_current = X_MAX


            left_lane_indsx.append(leftx_current)
            left_lane_indsy.append(lefty_current)
            right_lane_indsx.append(rightx_current)
            right_lane_indsy.append(righty_current)
            if  (DEBUG) & (TEST_MODE[0] == "IMAGE"):
                plt.plot(leftx_current, lefty_current, '*')
                plt.plot(rightx_current, righty_current, '*')
                plt.imshow(out_img)
                plt.show()

        # Extract left and right line pixel positions
        leftx = np.array(left_lane_indsx).astype(int)  # nonzerox[left_lane_inds]
        lefty = np.array(left_lane_indsy).astype(int)  # nonzeroy[left_lane_inds]
        rightx = np.array(right_lane_indsx).astype(int)  # nonzerox[right_lane_inds]
        righty = np.array(right_lane_indsy).astype(int)  # nonzeroy[right_lane_inds]


        #out_img[lefty, leftx] = [255,0,0]
        #out_img[righty, rightx] = [255,0,0]
        if not DEBUG & (TEST_MODE[0] == "IMAGE"):
            for center in zip(leftx, lefty):
                cv2.circle(out_img, center, 5, (255,255,0), thickness=3, lineType=12, shift=0)

            for center in zip(rightx, righty):
                cv2.circle(out_img, center, 5, (255,0,255), thickness=3, lineType=12, shift=0)

        #print(leftx, lefty)
        #print(rightx, righty)
        #print("writing ", leftx)
        self.leftx = leftx
        self.rightx = rightx
        return leftx, lefty, rightx, righty, out_img

    def _fit_polynomial(self, binary_warped=None):

        if binary_warped is None:
            binary_warped = self.binary_warped

        #if (self.__class__.count > 1) & (TEST_MODE[0] == "VIDEO") & (not (self.__class__.left_fit == [])) : #check if there is previous good fram
        if not self.caller.__class__.poly_resynch:
            return self.search_around_poly(binary_warped)

        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self._find_lane_pixels(binary_warped)

        order = 2
        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        left_fit = np.polyfit(lefty, leftx, order)
        right_fit = np.polyfit(righty, rightx, order)

        # Generate x and y values for plotting
        ploty = self.__class__.ploty #np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            self.left_fitx = left_fit[0] * ploty ** order + left_fit[1] * ploty**(order-1) + left_fit[2]* ploty**(order-2)  #+ left_fit[3]* ploty**(order-3)
            self.right_fitx = right_fit[0] * ploty ** order + right_fit[1] * ploty**(order-1) + right_fit[2]* ploty**(order-2)#+ right_fit[3]* ploty**(order-3)
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx = 1 * ploty ** 2 + 1 * ploty
            self.right_fitx = 1 * ploty ** 2 + 1 * ploty

        HANDLE = False
        if False :#order == 2:
            a = right_fit[0]
            b = right_fit[1]
            c = right_fit[2] - self.caller.img_size[0]
            d = b ** 2 - 4 * a * c
            y1 = (-b + math.sqrt(d)) / (2 * a)
            y2 = (-b - math.sqrt(d)) / (2 * a)

            print(y1, y2)

            YMAX = max(y1, y2)
            if (YMAX < self.caller.img_size[1]) :
                HANDLE = False
            else:
                HANDLE = True
                #ploty = np.linspace(YMAX + 1, self.caller.img_size[1] - 1)
                #self.left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            #self.right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


        ## Visualization ##
        # Colors in the left and right lane regions
        if (TEST_MODE[0] == "IMAGE"):
            try:
                out_img[ploty.astype(int), self.left_fitx.astype(int)] = [255, 0, 0]
                out_img[ploty.astype(int), self.right_fitx.astype(int)] = [0, 0, 255]

                # Plots the left and right polynomials on the lane lines
                plt.plot(self.left_fitx, ploty, color='red')
                plt.plot(self.right_fitx, ploty, color='red')
            except:
                pass

        if (not DEBUG) & (TEST_MODE[0] == "IMAGE"):
            try:
                plt.plot(self.right_fitx, ploty)
                plt.plot(self.left_fitx, ploty)
            #plt.legend()
            #plt.show()
            except:
                pass

        self.__class__.left_fit = left_fit
        self.__class__.right_fit = right_fit

        return out_img

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        self.window_height = window_height
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.leftx = leftx
        self.rightx = rightx

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped=None):
        if binary_warped is None:
            binary_warped = self.binary_warped

        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = self.ploty
        try:
            self.left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            self.right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx = 1 * ploty ** 2 + 1 * ploty
            self.right_fitx = 1 * ploty ** 2 + 1 * ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        if DEBUG:
            plt.plot(self.left_fitx, ploty, color='yellow')
            plt.plot(self.right_fitx, ploty, color='yellow')

        self.left_fit = left_fit
        self.right_fit = right_fit

        return out_img

    def project(self, annotate1="", annotate2="", pos1=0.5, pos2=0.5):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        #Saturation_image = self.caller.saturation.get(binary=True, masked=True)
        combined = self.caller.combined
        lane_image = combined
        lane_image = hough_lines(lane_image * 255, min_line_len=20, max_line_gap=1)
        color_image = np.dstack((lane_image, np.zeros_like(lane_image), np.zeros_like(lane_image)))
        color_nonzero = color_image.nonzero()
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.caller.__class__.Minv, (self.image.shape[1], self.image.shape[0]))
        # Combine the result with the original image
        temp_image = np.copy(self.image)
        temp_image[color_nonzero] = 0
        result = cv2.addWeighted(temp_image, 1, newwarp, 0.3, 0)
        if not USE_THREADING:
            fig0 = plt.figure(0)
            #plt.imshow(result).make_image()
            if 0:
                plt.annotate(annotate1, (10, 50), color='w', weight='bold',  size=14)
                plt.annotate(annotate2, (10, 100), color='w', weight='bold',  size=14)
                plt.imshow(result)

            #font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 40)

            font = ImageFont.truetype("arial.ttf", 40)
            img = Image.fromarray(result)
            draw = ImageDraw.Draw(img)
            draw.text((img.size[1]*pos1, 10), annotate1, (255, 255, 255), font=font)
            draw.text((img.size[1]*pos2, 80), annotate2, (255, 255, 255), font=font)
            draw = ImageDraw.Draw(img)
            #img.save("a_test.png")
            im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
            result = im_arr.reshape((img.size[1], img.size[0], 3))

        return result

    def measure_curvature_meter(self):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''
        #print(self.leftx)
        #if not list(self.leftx):
            #return 5555, 5555
        leftx = np.array(self.leftx[::-1])  # Reverse to match top-to-bottom in y
        rightx = np.array(self.rightx[::-1])  # Reverse to match top-to-bottom in y
        ploty = np.linspace(0, len(leftx) - 1, len(leftx))*self.caller.ysize/len(leftx)
        plotyy = ploty * self.ym_per_pix
        #print(1, leftx.size, rightx.size)
        left_fit = np.polyfit(plotyy, leftx * self.xm_per_pix, 2)
        right_fit = np.polyfit(plotyy, rightx * self.xm_per_pix, 2)
        #print(2, leftx , leftx.size)

        A = left_fit[0]
        B = left_fit[1]

        y = np.max(ploty * self.ym_per_pix)
        left_curverad = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / abs(
            2 * A)  ## Implement the calculation of the left line here

        A = right_fit[0]
        B = right_fit[1]
        right_curverad = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / abs(
            2 * A)  ## Implement the calculation of the right line here

        self.left_curverad = left_curverad
        self.right_curverad = right_curverad

        if TEST_MODE[0] == "IMAGE":
            print("Curves: ",left_curverad, right_curverad)
        return left_curverad, right_curverad

    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!
        #ploty, left_fit_cr, right_fit_cr = self.ploty, self.left_fit, self.right_fit
        leftx = np.array(self.leftx[::-1])  # Reverse to match top-to-bottom in y
        rightx = np.array(self.rightx[::-1])  # Reverse to match top-to-bottom in y
        ploty = np.linspace(0, len(leftx) - 1, len(leftx))*self.window_height
        plotyy = ploty * self.ym_per_pix

        plt.plot(leftx, ploty)
        plt.show()
        left_fit_cr = np.polyfit(plotyy, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(plotyy, rightx * self.xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** (3 / 2)) / abs(
            2 * left_fit_cr[0])  ## Implement the calculation of the left line here
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** (3 / 2)) / abs(
            2 * right_fit_cr[0])  ## Implement the calculation of the right line here

        return left_curverad, right_curverad

    def get_offset(self):
        # n =int(self.caller.ysize/2)
        Mid1 = self.caller.xsize/2
        Mid2 = (self.right_fitx[-1] + self.left_fitx[-1])/2
        #print(Mid1, Mid2, len(self.right_fitx))
        diff = float((Mid2 - Mid1)*self.xm_per_pix)

        # print(self.right_fitx[-1], self.left_fitx[-1],Mid1, Mid2, Mid1-Mid2 )
        #print(diff)
        return diff

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        if rightx.any():
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = self.right_fit

        if leftx.any():
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = self.left_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.left_fit = left_fit
        self.right_fit = right_fit

        return left_fitx, right_fitx, ploty

    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 50

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###

        # Generate x and y values for plotting
        # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        ploty = np.linspace(0, binary_warped.shape[0] - 1, 10)
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        left_fitx_low = left_fitx - margin  # boundry left one
        left_fitx_high = left_fitx + margin  # boundry left two

        right_fitx_low = right_fitx - margin  # boundry right one
        right_fitx_high = right_fitx + margin  # boundry right two

        left_lane_inds = None
        right_lane_inds = None

        plotyy = np.concatenate((ploty, ploty[::-1]), axis=None)
        plotxx = np.concatenate((left_fitx_low, left_fitx_high[::-1]), axis=None)
        zip_xx_yy = [list(zip(plotxx, plotyy))]
        vertices_left = np.array(zip_xx_yy, dtype=np.int32)
        image_interest_left = self.caller.region_of_interest(binary_warped, vertices_left)

        plotxx = np.concatenate((right_fitx_low, right_fitx_high[::-1]), axis=None)
        zip_xx_yy = [list(zip(plotxx, plotyy))]
        vertices_right = np.array(zip_xx_yy, dtype=np.int32)
        image_interest_right = self.caller.region_of_interest(binary_warped, vertices_right)

        if TEST_MODE[0] == "IMAGE":
            plt.imshow(binary_warped, 'gray')
            draw_points(*zip_xx_yy)
            plt.show()

            imgs = ["binary_warped", "image_interest_left", "image_interest_right"]
            img_dic = OrderedDict()
            for img in imgs:
                img_dic[img] = eval(img)
            draw_sub_plots(img_dic, col_no=3, raw_no=1, title="hh")

        # Again, extract left and right line pixel positions
        leftx = image_interest_left.nonzero()[1]
        lefty = image_interest_left.nonzero()[0]
        rightx = image_interest_right.nonzero()[1]
        righty = image_interest_right.nonzero()[0]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


        # Plot the polynomial lines onto the image
        if DEBUG:
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

        self.left_fitx =left_fitx
        self.right_fitx = right_fitx
        #print("write ", leftx.size, rightx.size)
        self.leftx = left_fitx
        self.rightx = right_fitx

        return result
