import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

#Part of this function was taken from the course material
def calculateCameraPoints(images, grid_m, grid_n):
    objWorldPoints = []  # 3D points in real world space
    imgPlanePoints = []  # 2D point in image plane

    # Board size 9x6
    # prepare object points, like [[0,0,0], [1,0,0], [2,0,0], .... , [8,5,0]]

    objPoints = np.zeros((grid_m*grid_n, 3), np.float32)
    objPoints[:, :2] = np.mgrid[0:grid_m, 0:grid_n].T.reshape(-1, 2)

    for imgName in images:
        # read each image
        img = mpimg.imread(imgName)
        # print(imgName)
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (grid_m, grid_n), None)

        # If corners are found, add object points, iamge points
        if ret == True:
            imgPlanePoints.append(corners)
            objWorldPoints.append(objPoints)

            # uncomment to save
            # plainName = imgName.split("/")[2]
           # draw and display the corners
            # img = cv2.drawChessboardCorners(
            # img, (grid_m, grid_n), corners, ret)

            # mpimg.imsave("../output_images/calibration/"+plainName, img)
           # plt.imshow(img)
           #     cv2.imshow('corners', cv2.cvtColor(
          #       img, cv2.COLOR_RGB2BGR))
           #    cv2.waitKey(0)

    return objWorldPoints, imgPlanePoints

#Part of this function was taken from the course material
def calcUndistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[1::-1], None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

#Part of this function was taken from the course material
def calcMtxDist(img, objpoints, imgpoints):
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[1::-1], None, None)
    return mtx, dist

#Part of this function was taken from the course material
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1],
                          corners[-1], corners[-nx]])
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
    return None, None

#Part of this function was taken from the course material
def lines_unwarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 100  # offset for dst points
    # Grab the image shape

    # For source points I'm grabbing the outer four detected corners
    a = [615, 435]
    b = [662, 435]
    c = [188, 720]
    d = [1118, 720]

    #cv2.fillConvexPoly(undist, np.array([a, b, d, c]), (255, 0, 0))
    # cv2.imwrite('../output_images/unwraped/elias.jpg', undist)

    src = np.float32([a, b, c, d])

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes

    offset = 310
    width = 600  # 610
    height = -1000  # -250
    dst = np.float32([
        [offset, height],
        [offset+width, height],
        [offset, 725],
        [offset+width, 725]])

    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    img_size = (gray.shape[1], gray.shape[0])
    warped = cv2.warpPerspective(undist, M, img_size)
    # Return the resulting image and matrix
    return warped, M


def test_distortion():
    # Read
    images = glob.glob('../camera_cal/calibration*.jpg')
    objpoints, imgpoints = calculateCameraPoints(images, 9, 6)
    # testImage = mpimg.imread(os.path.join(dirname, '../test_images/test1.jpg'))
    images = glob.glob('../test_images/straight_lines*.jpg')
    for imgName in images:
        img = cv2.imread(imgName)
        undistorted = calcUndistort(img, objpoints, imgpoints)
        plainName = imgName.split("/")[2]
        cv2.imwrite('../output_images/undistorted/'+plainName, undistorted)
    # cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))
    # cv2.imshow('undistorted', undistorted)
   # cv2.waitKey(0)


def test_unwarp_squares():
    # Read
    images = glob.glob('../camera_cal/calibration*.jpg')
    nx = 9
    ny = 6
    objpoints, imgpoints = calculateCameraPoints(images, nx, ny)
    isFirst = True
    mtx = None
    dist = None
    for imgName in images:
        print(imgName)
        img = cv2.imread(imgName)
        if isFirst:
            mtx, dist = calcMtxDist(img, objpoints, imgpoints)
            print(mtx)
            print(dist)
            isFirst = False
       # cv2.imshow('image2', image)
      #  cv2.waitKey(0)
        unwarped, _ = corners_unwarp(np.copy(img), nx, ny, mtx, dist)
        if unwarped is not None:
            plainName = imgName.split("/")[2]
            cv2.imwrite('../output_images/unwarped/'+plainName, unwarped)


def test_unwarp():
    # Read
    images = glob.glob('../camera_cal/calibration*.jpg')
    nx = 9
    ny = 6
    objpoints, imgpoints = calculateCameraPoints(images, nx, ny)
    isFirst = True
    mtx = None
    dist = None
    images = glob.glob('../test_images/test*.jpg')
    for imgName in images:
        print(imgName)
        img = cv2.imread(imgName)
        if isFirst:
            mtx, dist = calcMtxDist(np.copy(img), objpoints, imgpoints)
            print(mtx)
            print(dist)
            isFirst = False
       # cv2.imshow('image2', image)
      #  cv2.waitKey(0)
        unwarped, _ = lines_unwarp(np.copy(img), mtx, dist)
        if unwarped is not None:
            plainName = imgName.split("/")[2]
            cv2.imwrite('../output_images/unwarped/'+plainName, unwarped)


# test_unwarp()

# test_distortion()

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(testImage)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
