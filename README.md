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


## Pipeline (Single Images)

### 1. Examples of a Distortion-corrected Images

### 2. Image Thresholds

### 3. Perspective Transform

### 4. Lane Lines Detection

### 5. Lane Curvature and Vehicle Position with respect to Lane Center

### 6. Final Image Pipeline 

## Pipeline (Video)


Here's a link to my [video](output_video.mp4) result

## Discussion

