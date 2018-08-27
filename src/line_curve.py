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


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids


def convolution(warped):
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 80 # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        l_non_zeros = l_points.nonzero()
        r_non_zeros = r_points.nonzero()

        leftx = l_non_zeros[0]
        lefty = l_non_zeros[1]
        rightx = r_non_zeros[0]
        righty = r_non_zeros[1]

        return lefty, leftx , righty,rightx
    # If no window centers found, just display orginal road image
    else:
        return None, None, None,None


class Line():
    n = 30

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
        # print("Poly: ", poly)
        print("Average curvature: ", self.avgCurvature)
        self.diffs = np.subtract(poly,self.current_fit)
        # print("poly difs: ", self.diffs)
        self.current_fit = poly

        if detected:
            self.detected = True

            if self.n == 1:
                self.recent_xfitted = []
                self.polys = []
            else:
                self.recent_xfitted = self.recent_xfitted[-self.n+1:]
                self.polys = self.polys[-self.n+1:]

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

        if self.radius_of_curvature is not None:
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
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

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




class Params():
    xm_per_pix = 3.7/700
    ym_per_pix = 30/720
    min_pixels = 400
    def __init__(self, ploty, leftx, lefty, rightx, righty, shape):
        if len(leftx) > self.min_pixels and len(rightx) > self.min_pixels:
            left_fit, right_fit, _, _ = fit_poly(ploty, leftx, lefty, rightx, righty, self.ym_per_pix, self.xm_per_pix)
            left_curverad, right_curverad = measure_curvature_real(ploty, left_fit, right_fit, self.ym_per_pix, self.xm_per_pix)
            left_fit, right_fit, left_fitx, right_fitx = fit_poly(ploty, leftx, lefty, rightx, righty)
            roadCenter = (left_fitx[-1] +right_fitx[-1])/2
            carCenter = (shape[1]/2)
            delta = round((carCenter - roadCenter) * self.xm_per_pix, 3)
            deltaLeft = (carCenter - left_fitx[-1])*self.xm_per_pix
            deltaRight = (right_fitx[-1] - carCenter)*self.xm_per_pix
            self.left_fit = left_fit
            self.right_fit = right_fit
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx
            self.left_curverad = left_curverad
            self.right_curverad = right_curverad
            self.delta = delta
            self.delta_left = deltaLeft
            self.delta_right = deltaRight
            self.width_by_image_width= (right_fitx[-1] - left_fitx[-1]) / shape[1]
            self.is_valid = True
        else:
            self.is_valid = False
        
        # Avoids an error if the above is not implemented fully

    def isDetected(self):
        detected = self.is_valid and self.width_by_image_width < 0.7 and abs(1 - self.delta_left / self.delta_right) < 0.15 and ( abs(abs(self.left_fit[1]) / abs(self.right_fit[1]) - 1) < 1 or abs(1- self.left_curverad / self.right_curverad ) < 1 ) #and ( left.best_fit is None or (abs(1 - abs(self.left_fit[0]) / abs(left.best_fit[0])) < 1  and abs(1 - abs(self.right_fit[0]) / abs(right.best_fit[0])) < 1 ) )
        return detected
    def log(self):
        if self.is_valid is True:
            print("delta left ", self.delta_left)
            print("delta right ", self.delta_right)
            print("left curve ", self.left_curverad)
            print("right curve ", self.right_curverad)
            print("left_fit[0] ", self.left_fit[0])
            print("right_fit[0] ", self.right_fit[0])
            print("left_fit[1] ", self.left_fit[1])
            print("right_fit[1] ", self.right_fit[1])
        else:
            print("Object is not valid")


def lane_finding_pipeline(img):
    c = Calibration()
    ploty = c.getPloty()
    mtx, dist = c.calcMtxDist()

    img_thrsh = threshold_pipeline(img)
    unwarped, M = lines_unwarp(img_thrsh, mtx, dist)
    
    binary_unwarped = unwarped[:, :, 0]

    leftx, lefty, rightx, righty = None, None, None, None

    if left.detected and right.detected:
        # print("detected :)")
        leftx, lefty, rightx, righty = search_around_poly(binary_unwarped, left.best_fit, right.best_fit)
    else:
        # print("not detected :(")
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_unwarped)

    p = Params(ploty, leftx, lefty, rightx, righty, img.shape)

    detected = p.isDetected()
    if detected:
        print("Detected :)")
    else:
        p.log()
        # retry by finding the pixels from beginning
        leftx, lefty, rightx, righty = convolution(binary_unwarped)
        p = Params(ploty, leftx, lefty, rightx, righty, img.shape)
        detected = p.isDetected()
        if detected:
            print("===> Conv detected :)")
        else:
            p.log()
            leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_unwarped)
            p = Params(ploty, leftx, lefty, rightx, righty, img.shape)
            detected = p.isDetected()
            if detected:
                print("===> Retry Detected :)")

    left.update(p.left_fitx, p.left_fit, p.left_curverad, p.delta_left, leftx, lefty, detected)
    right.update(p.right_fitx, p.right_fit, p.right_curverad, p.delta_right, rightx, righty, detected)

    left_fitx = p.left_fitx
    right_fitx = p.right_fitx
    left_curverad = p.left_curverad
    right_curverad = p.right_curverad

    if left.bestx is not None and right.bestx is not None:
        left_fitx = left.bestx
        right_fitx = right.bestx
        left_curverad = left.radius_of_curvature
        right_curverad = right.radius_of_curvature

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

    lane_lines[left.ally, left.allx] = [255, 0, 0]
    lane_lines[right.ally, right.allx] = [0, 0, 255]

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
    if p.delta > 0:
        txt = str(abs(p.delta)) + "m right"
    else:
        txt = str(abs(p.delta)) + "m left"
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
            print("Calibrating...")
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

def pipeline_on_images():
    images = glob.glob('../test_images/test*.jpg')
    for imgName in images:
        print(imgName)
        left.detected = False
        right.detected = False
        plainName = imgName.split("/")[2]
        img = cv2.imread(imgName)
        result = lane_finding_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite('../output_images/lane_finding_pipeline/'+plainName, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

class Counter():
    def __init__(self):
        self.count = 0
    def advance(self):
        self.count = self.count + 1
        return str(self.count)

count = Counter()
def process_image2(image):
    if count.count < 10:
        cv2.imwrite('../challenge_images/'+ count.advance() + '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image


threshAndTransform()

left.n = 1
right.n = 1
left = Line()
right = Line()
pipeline_on_images()
left = Line()
right = Line()


video_output = '../output_videos/project_video.mp4'
clip1 = VideoFileClip("../project_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(video_output, audio=False)

left = Line()
right = Line()


video_output = '../output_videos/challenge_video.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(video_output, audio=False)


left = Line()
right = Line()

video_output = '../output_videos/harder_challenge_video.mp4'
clip1 = VideoFileClip("../harder_challenge_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(video_output, audio=False)


