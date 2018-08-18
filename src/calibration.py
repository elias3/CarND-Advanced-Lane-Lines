import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


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
            #plainName = imgName.split("/")[2]
           # draw and display the corners
            # img = cv2.drawChessboardCorners(
            # img, (grid_m, grid_n), corners, ret)

            #mpimg.imsave("../output_images/calibration/"+plainName, img)
           # plt.imshow(img)
           #     cv2.imshow('corners', cv2.cvtColor(
          #       img, cv2.COLOR_RGB2BGR))
           #    cv2.waitKey(0)

    return objWorldPoints, imgPlanePoints


def calcUndistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[1::-1], None, None)

    dst = cv2.undistort(np.copy(img), mtx, dist, None, mtx)

    return dst


def test_distortion():
    # Read
    images = glob.glob('../camera_cal/calibration*.jpg')
    objpoints, imgpoints = calculateCameraPoints(images, 9, 6)
    # testImage = mpimg.imread(os.path.join(dirname, '../test_images/test1.jpg'))
    images = glob.glob('../test_images/test*.jpg')
    for imgName in images:
        img = cv2.imread(imgName)
        undistorted = calcUndistort(img, objpoints, imgpoints)
        plainName = imgName.split("/")[2]
        cv2.imwrite('../output_images/undistorted/'+plainName, undistorted)
    # cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))
    #cv2.imshow('undistorted', undistorted)
   # cv2.waitKey(0)


# test_distortion()

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(testImage)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
