# Advanced Lane Finding Project
## Overview
In this project, a software pipeline with the use of computer vision techniques is written to identify the lane boundaries in a video from a front-facing camera on a car. The output of the pipeline is a video with lane lines identified and estimation of road curvature as well as vehicle location with respect to the center of the lane. 

<img src="output_images/image_out_8.jpg" width="75%" height="75%">

### The Goals of this Project:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Project Files
The project includes the following files:
* `README.md` - A markdown file explaining the project structure and training approach
* `image_undist.py` - Script used for camera calibration and distortion correction
* `image_thresholds.py` - Script used for color and Sobel gradients transformation to create a thresholded binary image
* `perspective_transform.py` - Script used to apply a perspective transform ("birds-eye view") 
* `lane_lines.py` - Script with function to detect lane lines, lane curvature and vehicle position
* `image_pipeline.py` - Image processing pipeline to find the lane lines in images
* `video_pipeline.py` - Image processing pipeline to find the lane lines in video frames
* `output_video.mp4` - A video recording with lane lines identified

## Camera Calibration
The first step is to correct the distorsion of images by calibrating the camera. Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image; this transformation isn’t perfect. Distortion actually changes what the shape and size of these 3D objects appear to be. So, the first step in analyzing camera images, is to undo this distortion so that you can get correct and useful information out of them. Without that the apparent size, shape of an object in an image is incorrect.

There are two main steps to this process: use chessboard images to obtain image points and object points, and then use the OpenCV functions `cv2.calibrateCamera()` and `cv2.undistort()` to compute the calibration and undistortion.


The code as well as its description is saved in the `image_undist.py` file. The file contains two functions:
* `cam_calibration()` - Calculates object points, image points and performs camera calibration parameters
* `image_undistort()` - Performs image distortion correction and returns the undistorted image 


Here is how to run the Python script that calculates camera matrix, distortion coefficients from calibration images and undistort the images:

``` 
>python image_undist.py -h
usage: image_undist.py [-h] [-p PATH]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  specify path to calibration image files (without -p
                        the default is: ./camera_cal/calibration*.jpg)
```
To specify your own path to calibration images, simply use wildcard symbol `*.jpg` to read all jpg images as follows
```
>python image_undist.py -p ./your_folder/*.jpg
```

Here is a result of applying `image_undist.py` on an image 

<img src="output_images/01_dist_undistorted_image.png" width="100%" height="100%">

## Pipeline (Single Images)

### 1. Examples of a Distortion-corrected Images
Here is an example of applying the camera calibration parameters on test images in order to undistort them. 

<img src="output_images/02_dist_undistorted_test_image.png" width="100%" height="100%">

One can observe that there is a difference between distored (left column) and undistored (right column) images especialy apparent close to the edges showing that the distortion was corrected. These images will be taken as input to the image and video pipeline.

### 2. Image Thresholds

### 3. Perspective Transform

<img src="output_images/04_perspective_transform.png" width="100%" height="100%">


### 4. Lane Lines Detection

### 5. Lane Curvature and Vehicle Position with respect to Lane Center

### 6. Final Image Pipeline 

## Pipeline (Video)


Here's a link to my [video](output_video.mp4) result

## Discussion

