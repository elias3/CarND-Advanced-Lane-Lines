import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

from color_gradient_thrsh import threshold_pipeline
from calibration import calculateCameraPoints,calcMtxDist,lines_unwarp


def threshAndTransform():
    # Read calubration images:
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
        img_thrsh = threshold_pipeline(img)
        unwarped, _ = lines_unwarp(img_thrsh, mtx, dist)
        if unwarped is not None:
            plainName = imgName.split("/")[2]
            cv2.imwrite('../output_images/pipeline/'+plainName, unwarped)

threshAndTransform()