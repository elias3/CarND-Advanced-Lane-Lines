import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from moviepy.editor import VideoFileClip


from color_gradient_thrsh import threshold_pipeline
from calibration import calculateCameraPoints, calcMtxDist, lines_unwarp


class Line():
    n = 15
    min_number_pixels = 6000

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the last n fits
        self.polys = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.count = 0
        self.avgCurvature = 0

    def update(self, xfitted, poly, radius, line_pos, allx, ally, detected):

        print("Poly: ", poly)
        print("Average curvature: ", self.avgCurvature)
        self.diffs = np.subtract(poly,self.current_fit)
        print("poly difs: ", self.diffs)
        self.current_fit = poly

        if detected:
            self.detected = True

            self.recent_xfitted = self.recent_xfitted[-self.n+1:]
            self.recent_xfitted.append(xfitted)
            self.bestx = np.average(self.recent_xfitted, axis = 0)

            self.polys = self.polys[-self.n+1:]
            self.polys.append(poly)

            self.best_fit = np.average(self.polys, axis = 0)
            self.allx = allx
            self.ally = ally   
            self.radius_of_curvature = radius
        else:
            self.detected = False

        sum = self.avgCurvature * self.count
        self.count = self.count + 1
        self.avgCurvature = (sum + self.radius_of_curvature) / self.count




def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
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
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
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

    return leftx, lefty, rightx, righty, out_img

def fit_poly(ploty, leftx, lefty, rightx, righty, ym_per_pix = 1, xm_per_pix = 1):
    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Generate x and y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, left_fitx, right_fitx

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 80

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped, saveFilePath=None, ym_per_pix = 1, xm_per_pix = 1) :
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Generate x and y values for plotting
    ploty = Calibration().getPloty()

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty


    if saveFilePath is not None:
        # Plots the left and right polynomials on the lane lines
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        plt.close()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig(saveFilePath)
        plt.close()

    return left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty


def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


left = Line()
right = Line()

def lane_finding_pipeline(img):
    c = Calibration()
    ploty = c.getPloty()
    mtx, dist = c.calcMtxDist()
    img_thrsh = threshold_pipeline(img)
    unwarped, M = lines_unwarp(img_thrsh, mtx, dist)
    
    binary_unwarped = unwarped[:, :, 0]

    leftx, lefty, rightx, righty = None, None, None, None
    
    # if left.detected and right.detected:
        # print("detected :)")
        # leftx, lefty, rightx, righty = search_around_poly(binary_unwarped, left.best_fit, right.best_fit)
    # else:
        # print("not detected :(")
    leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_unwarped)

    xm_per_pix = 3.7/700
    ym_per_pix = 30/720
    left_fit, right_fit, _, _ = fit_poly(ploty, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit, right_fit, ym_per_pix, xm_per_pix)
    left_fit, right_fit, left_fitx, right_fitx = fit_poly(ploty, leftx, lefty, rightx, righty)


    roadCenter = (left_fitx[-1] +right_fitx[-1])/2
    carCenter = (img.shape[1]/2)
    delta = round((carCenter - roadCenter) * xm_per_pix, 3)

    deltaLeft = (carCenter - left_fitx[-1])*xm_per_pix
    deltaRight = (right_fitx[-1] - carCenter)*xm_per_pix

    # print("delta left ", deltaLeft)
    # print("delta right ", deltaRight)
    # print("left curve ", left_curverad)
    # print("right curve ", right_curverad)
    # print("left_fit[0] ", left_fit[0])
    # print("right_fit[0] ", right_fit[0])
    # print("left_fit[1] ", left_fit[1])
    # print("right_fit[1] ", right_fit[1])

    detected = abs(1 - deltaLeft / deltaRight) < 0.6 and abs(left_fit[1] / right_fit[1] - 1) < 0.8
    if detected:
        print("Detected :)")
    else:
        print("Not Detected :(")

    left.update(left_fitx, left_fit, left_curverad, deltaLeft, leftx, lefty, detected)
    right.update(right_fitx, right_fit, right_curverad, deltaRight, rightx, righty, detected)

    if not detected and left.bestx is not None and right.bestx is not None:
        left_fitx = left.bestx
        right_fitx = right.bestx
        left_curverad = left.radius_of_curvature
        right_curverad = right.radius_of_curvature


    # cv2.imwrite('../output_images/pipeline/'+plainName, out_img)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_unwarped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    lane_lines = np.copy(color_warp)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    lane_lines[lefty, leftx] = [255, 0, 0]
    lane_lines[righty, rightx] = [0, 0, 255]


    # left_lane = np.concatenate((left_fitx[::-1], axis=0),np.concatenate(ploty, axis=0))
    
    #cv2.fillPoly(color_warp,[left_lane],color=[255,0,0])
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, inv(M), (img.shape[1], img.shape[0]))

    lane_lines_warped = cv2.warpPerspective(
        lane_lines, inv(M), (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    combined = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(combined, 1, lane_lines_warped, 1, 0)

    curvature = str(int((left_curverad + right_curverad) / 2))
    cv2.putText(result,'Radius of Curvature = ' + curvature + '(m)',(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)


    txt = ""
    if delta > 0:
        txt = str(abs(delta)) + "m right"
    else:
        txt = str(abs(delta)) + "m left"
  
  
    cv2.putText(result,'Vehicle is '+txt+' of center',(100,170), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
    return result


def process_image(image):
    return lane_finding_pipeline(image)


class Calibration:
    class __impl:
        def __init__(self):
            images = glob.glob('../camera_cal/calibration*.jpg')
            nx = 9
            ny = 6
            objpoints, imgpoints = calculateCameraPoints(
                images, nx, ny)
            img = cv2.imread('../test_images/test1.jpg')
            self.mtx, self.dist = calcMtxDist(
                img, objpoints, imgpoints)

            self.ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        def calcMtxDist(self):
            return self.mtx, self.dist

        def getPloty(self):
            return self.ploty


    __instance = __impl()

    def __getattr__(self, attr):
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__instance, attr, value)

def threshAndTransform():
    # Read calubration images:
    # images = glob.glob('../camera_cal/calibration*.jpg')
    # nx = 9
    # ny = 6
    # objpoints, imgpoints = calculateCameraPoints(images, nx, ny)
    # isFirst = True
    # mtx = None
    # dist = None
    c = Calibration()
    ploty = c.getPloty()
    mtx, dist = c.calcMtxDist()
    print(mtx)
    print(dist)
    images = glob.glob('../test_images/test*.jpg')
    for imgName in images:
        print(imgName)
        img = cv2.imread(imgName)
        plainName = imgName.split("/")[2]
        img_thrsh = threshold_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite('../output_images/threshold/'+plainName, img_thrsh)
        unwarped, M = lines_unwarp(img_thrsh, mtx, dist)
        cv2.imwrite('../output_images/unwarped_thresh/'+plainName, unwarped)

        left_fit_cr, right_fit_cr, _, _, _, _, _, _ = fit_polynomial(
        unwarped[:, :, 0], ym_per_pix = 30/720, xm_per_pix = 3.7/700)

        left_curverad, right_curverad = measure_curvature_real(
            ploty, left_fit_cr, right_fit_cr)
        print(left_curverad, right_curverad)

        _, _, left_fitx, right_fitx, _, _, _, _ = fit_polynomial(
            unwarped[:, :, 0], '../output_images/pipeline/'+plainName)
        # cv2.imwrite('../output_images/pipeline/'+plainName, out_img)


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(unwarped[:, :, 0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp, inv(M), (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        #txt = left_curverad, right_curverad
        curvature = str(int(min(left_curverad, right_curverad)))
        cv2.putText(result,'Radius of Curvature = ' + curvature + '(m)',(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
        # cv2.putText(result,'Vehicle is 0.17m left of center',(100,170), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite('../output_images/transformed_txt/'+plainName, result)

        # plt.imshow(result)
# threshAndTransform()

def pipeline_on_images():
    images = glob.glob('../test_images/test*.jpg')
    for imgName in images:
        print(imgName)
        plainName = imgName.split("/")[2]
        img = cv2.imread(imgName)
        result = lane_finding_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite('../output_images/lane_finding_pipeline/'+plainName, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


pipeline_on_images()

# video_output = '../output_videos/project_video.mp4'
# clip1 = VideoFileClip("../project_video.mp4")
# project_clip = clip1.fl_image(process_image)
# project_clip.write_videofile(video_output, audio=False)
