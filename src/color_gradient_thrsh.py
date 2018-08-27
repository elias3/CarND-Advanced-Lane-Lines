import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob


# Part of this function was taken from the course material
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to imgx
    # 1) Convert to hls
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(l_channel, cv2.CV_64F, int(
        orient == 'x'), int(orient == 'y'), ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def threshold_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), r_thresh=(200, 255), g_thresh=(180, 255), l_thresh=(220, 255), ksize=9):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    r_channel = img[:, :, 0]
    s_channel = hls[:, :, 2]

    light_channel = lab[:, :, 0]
    # Sobel x
    sxbinary = abs_sobel_thresh(
        img, orient='x', sobel_kernel=ksize, thresh=sx_thresh)

    sybinary = abs_sobel_thresh(
        img, orient='y', sobel_kernel=ksize, thresh=sx_thresh)

    combined = np.zeros_like(s_channel)

    red_thresh = (r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])
    sat_thresh = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    gray_thresh = (gray >= g_thresh[0]) & (gray <= g_thresh[1])
    light_thres = (light_channel >= l_thresh[0]) & (
        light_channel <= l_thresh[1])

    combined[((light_thres | sxbinary == 1) | (
        sat_thresh & gray_thresh)) & (red_thresh | sybinary == 1)] = 1
    return np.dstack((combined, combined, combined)) * 255


def test():
    images = glob.glob('../challenge_images/*.jpg')
    for imgName in images:
        img = mpimg.imread(imgName)
        thresh = threshold_pipeline(img)
        plainName = imgName.split("/")[2]
        cv2.imwrite('../output_images/challenge_images/'+plainName, thresh)


#test()
