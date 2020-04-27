import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

from image_undist import cam_calibration, image_undistort
from image_thresholds import thresholds
from perspective_transform import image_warp
from lane_lines import Line, sliding_window, draw_lane_lines, draw_lane_polygon, \
                       measure_curvature_real, vehicle_position


def pipeline(img):
    
    # Initialize two instance of class Line() to represent left and right lane lines
    left_line  = Line()
    right_line = Line()
    
    # Camera calibration: returns the camera matrix and distortion coefficients
    mtx, dist = cam_calibration(glob.glob('./camera_cal/calibration*.jpg'))
        
    # Perform image distortion correction
    undistorted = image_undistort(img, mtx, dist)
    
    # Calculate combination of color and gradient thresholds
    combined_binary = thresholds(undistorted)
    
    # Calculate perspective transform
    warped_img = image_warp(undistorted, reversed=False, box=False)

    # Calculate reverse perspective transform
    unwarped_img, Minv = image_warp(undistorted, reversed=True, box=False)

    # Calculate perspective transform of binary image 
    combined_binary_warped = image_warp(combined_binary)
        
    # Detect lane lines
    left_fit, right_fit, out_image = sliding_window(combined_binary_warped)        
    
    # Calculate the lane curvature
    left_lane = measure_curvature_real(combined_binary_warped.shape, left_fit)
    right_lane = measure_curvature_real(combined_binary_warped.shape, right_fit)
    lane_curvature = (left_lane + right_lane) / 2
    
    # Calculate vehicle offset
    center_offset = vehicle_position(combined_binary_warped.shape, left_fit, right_fit)
    
    # Draw lane polygon
    result = draw_lane_polygon(undistorted, warped_img, left_fit, right_fit, Minv,
                               lane_curvature, center_offset)

    return result
    
    
if __name__ == '__main__':
    
    import argparse
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default='test_images/*.jpg',
                        help="specify path with image file name as PATH \
                        (without -p the default is: %(default)s)")
    args = parser.parse_args()
    print("Path to File(s) set to: {}".format(args.path))
    
    # Read in images from path given by args.path
    image_path = glob.glob(args.path)
    print(image_path)
    
    for i, image in enumerate(image_path):
        print('Processing Image: {}'.format(image))
        img_rgb = cv2.imread(image, cv2.COLOR_BGR2RGB)
        result = pipeline(img_rgb)
        cv2.imwrite('output_images/test_images/image_out_{}.jpg'.format(i+1), result)

        
        
        
        
        
        
        