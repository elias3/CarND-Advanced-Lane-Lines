import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to imgx
    # 1) Convert to grayscale
   # gray = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
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
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    # 6) Return this mask as your binary_output image
   # binary_output = np.copy(img) # Remove this line
    return binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
   # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobel_x**2+sobel_y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a binary mask where mag thresholds are met
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
   # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    binary = np.zeros_like(direction)
    binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), r_thresh=(200, 255), g_thresh=(180, 255), l_thresh=(220, 255), ksize=5):
    img = np.copy(img)
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    r_channel = img[:, :, 0]
    s_channel = hls[:, :, 2]

    light_channel = lab[:,:,0]
    # Sobel x
    sxbinary = abs_sobel_thresh(
        img, orient='x', sobel_kernel=15, thresh=sx_thresh)

    combined = np.zeros_like(s_channel)

    red_thresh = (r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])
    sat_thresh = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    gray_thresh = (gray >= g_thresh[0]) & (gray <= g_thresh[1])
    light_thres = (light_channel >= l_thresh[0]) & (light_channel <= l_thresh[1])

    combined[  ((light_thres | sxbinary==1) | (sat_thresh & gray_thresh)) & (red_thresh)  ] = 1

    # combined[((r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])) | ((s_channel >= s_thresh[0]) & (
    #     s_channel <= s_thresh[1])) & ()] = 1
    #   & (r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])) | ((sxbinary == 1) & (sybinary == 1))
    # Stack each channel
    return np.dstack((combined, combined, combined)) * 255
    # return np.dstack((np.zeros_like(s_channel), combined, sxbinary)) * 255

   # return np.dstack((r_binary, s_binary, sxbinary)) * 255
   # color_binary = np.dstack(
   #     (np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
   # return color_binary



images = glob.glob('../test_images/test*.jpg')
for imgName in images:
    img = mpimg.imread(imgName)
    thresh = pipeline(img)
    plainName = imgName.split("/")[2]
    cv2.imwrite('../output_images/threshold/'+plainName, thresh)

# image = mpimg.imread('../test_images/test1.jpg')
# result = pipeline(image)
# cv2.imshow('pipelne', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

# image = mpimg.imread('../test_images/test2.jpg')
# result = pipeline(image)
# cv2.imshow('pipelne', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)


# image = mpimg.imread('../test_images/test3.jpg')
# result = pipeline(image)
# cv2.imshow('pipelne', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

# image = mpimg.imread('../test_images/test4.jpg')
# result = pipeline(image)
# cv2.imshow('pipelne', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

# image = mpimg.imread('../test_images/test5.jpg')
# result = pipeline(image)
# cv2.imshow('pipelne', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

# image = mpimg.imread('../test_images/test6.jpg')
# result = pipeline(image)
# cv2.imshow('pipelne', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)


# cv2.imshow(result)
