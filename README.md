## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calib_input]: ./camera_cal/calibration7.jpg "Distroted"
[calib_iden]: ./output_images/calibration/calibration7.jpg "Checkerboxes identified"
[calib_warp]: ./output_images/unwarped/calibration7.jpg "Warped"
[calib_undist]: ./output_images/undistorted/calibration7.jpg "Undistored"
[pipeline_undist]: ./output_images/undistorted/test1.jpg "Pipeline Undistored"
[threshold]: ./output_images/threshold/test3.jpg "Thresholding"
[curvature]: ./examples/curvature_formula.svg "Curvature Radius Formula"
[image1]: ./examples/test1.png "Input"
[image1_thresh_transformed]: ./output_images/unwarped_thresh/test1.jpg
[image1_transformed]: ./output_images/lane_finding_pipeline/test1.jpg
[image2]: ./test_images/test1.jpg "Road Transformed"
[thresh_detected]: ./output_images/pipeline/test1.jpg "Histogram + sliding window"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #11 through #48 of the file called `calibration.py`.  

![alt text][calib_input]

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objPoints` is just a replicated array of coordinates, and `objWorldPoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgPlanePoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][calib_iden]

I then used the output `objWorldPoints` and `imgPlanePoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calib_warp]
![alt_text][calib_undist]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][pipeline_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, the filters/channels that I used are the following: sobelx, sobely, gray, light (from lab), saturation from hls, red. (thresholding steps at lines #31 through #61 in `color_gradient_thrsh.py`).  Here's an example of my output for this step.

![alt text][threshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `lines_unwarp()`, which appears in lines 111 through 154 in the file `calibration.py` (src/calibration.py) .  The `lines_unwarp()` function takes as inputs an image (`img`), as well as (`mtx`) and dist (`dst`) - the output of `calibrateCamera()`.  I chose the hardcode the source and destination points in the following manner:

```python
a = [615, 435]
b = [662, 435]
c = [188, 720]
d = [1118, 720]
src = np.float32([a, b, c, d])    
offset = 310
width = 600  
height = -1000  
dst = np.float32([
    [offset, height],
    [offset+width, height],
    [offset, 725],
    [offset+width, 725]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 615, 435      | 310, -1000    |
| 662, 435      | 910, -1000    |
| 188, 720      | 310, 720      |
| 1118, 720     | 1718, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image1_thresh_transformed]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I took the histogram of the bottom half of the image and calculated the midpoint of the width.
Everything below this midpoint will be part of the search of the left line and everything that is higher is part of the search for the right line.

I then detected the non-zero pixels in the image, i.e. the ones that are white in the binary image. The detection is done in a window of pixels and then moving to the next window while adjusting to the left/right based on the amount of pixels. This is done in `find_lane_pixels()`.

![alt text][thresh_detected]

Then I fitted the the detected lines with second order polynomials.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #329 through #348 in my code in `line_curve.py`. This calculation is based on the following formula:

![alt text][curvature]

In our case y is the second order polynomial. And its derivative is just: 2Ay + B
The second derivative will be: 2A*y.
In our case the function is f(y) and not f(x) as in the formula. This is due to the reason that the lines can be parallel, and in an f(x) representation it will no longer be a function, since the same x can have multiple values. 

For the position of the vehicle, I assumed that the car is at the center of the image. Then I calculated the center of the road using the difference of the two x coordinates of the two detected polynomials - they are fond at the end of the array since we are searching for the highest y value.

I did this in lines #329 through #348 in my code in `line_curve.py`. This calculation is based on the following formula:

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #363 through #367 in my code in `line_curve.py` in the class `Params()` in the constructor.

![alt text][image1_transformed]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[link to my video result](./output_videos/project_video.mp4)

[link to my challenge video result](./output_videos/challenge_video.mp4)

[link to my harder challenge video result](./output_videos/harder_challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took relies on detecting the lines using a histogram with a moving window algorithm. For the next frame, I use the output polynomial as the basis for detecting the region. When this fails I invoke the convultion approach. This multitude of approches is costly and should be optimized. Since, in real world scenarios we can't affoard that much of processing.

Another issue that I faced is that different thresholding yield to different results. So it might be good to discover a dynamic thresholding approach to help detection under different lightnining conditions.

Also, I didn't take into consideration that only one of the lines might be detected for a long period. Which meant that I had trouble recognizing part of the segment in the harder challenge video.

I had ideas of training the thresholding parameters using a neural network. Also experimenting with different windows sizes and thresholds for the convolution.
