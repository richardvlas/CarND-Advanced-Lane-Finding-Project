import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

from image_undist import cam_calibration, image_undistort
from image_thresholds import thresholds
from perspective_transform import image_warp
from lane_lines import Line, sliding_window, search_around_poly, draw_lane_lines, draw_lane_polygon, measure_curvature_real, vehicle_position, avg_fit, validate


# Initialize two instance of class Line() to represent left and right lane lines
left_line  = Line()
right_line = Line()


def pipeline(img): 
    
    # Camera calibration: returns the camera matrix and distortion coefficients
    mtx, dist = cam_calibration(glob.glob('./camera_cal/calibration*.jpg'))
        
    # Perform image distortion correction
    undistorted = image_undistort(img, mtx, dist)
    
    # Calculate combination of color and gradient thresholds
    combined_binary = thresholds(undistorted)
    
    # Calculate perspective transform
    #warped_img = image_warp(undistorted, reversed=False, box=False)

    # Calculate perspective transform of binary image 
    combined_binary_warped = image_warp(combined_binary)
    
    # Calculate reverse perspective transform
    unwarped_img, Minv = image_warp(undistorted, reversed=True, box=False)

        
    # Detect lane lines
    # If during last iteration lane lines were detected -> use search_around_poly
    # Otherwise use sliding window search
    if (left_line.detected == False) or (right_line.detected == False):
                
        try:
            # Use Sliding Window Lane lines finding algorithm
            print('**Sliding Window Activated !')
            left_fit, right_fit, out_image = sliding_window(combined_binary_warped)
        
        except Exception as ex:
            print('**Sliding Window Algorithm Failed use previous fit !')                
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            
            left_fit = left_line.previous_fit  
            right_fit = right_line.previous_fit
    else:
        try:
            # Use search_around_poly algorithm
            print('**Search Around Activated !')
            left_fit, right_fit, out_image = search_around_poly(combined_binary_warped, 
                                                                left_line.previous_fit, 
                                                                right_line.previous_fit)
        except TypeError:
            try:
                # Use Sliding Window again if search around fails!
                print('**Search Around Failed, returing back to Sliding windowActivated !')
                left_fit, right_fit, out_image = sliding_window(combined_binary_warped)

            except Exception as ex:
                print('**Both Algorithms Failed use previous fit !')                
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                left_fit = left_line.previous_fit
                right_fit = right_line.previous_fit
    
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # Sanity Check to validate that lane lines are detected properly
    validate(combined_binary_warped.shape, left_line, right_line)
    
    # Calculate average polynomial coefficients for previous n iterations 
    # Store in self.n_fits
    left_line.average_fit = avg_fit(left_line, n=3)
    right_line.average_fit = avg_fit(right_line, n=3)
            
    # Calculate the lane curvature
    left_lane_curverad = measure_curvature_real(combined_binary_warped.shape, 
                                                left_line.average_fit)
    right_lane_curverad = measure_curvature_real(combined_binary_warped.shape, 
                                                 right_line.average_fit)
    lane_curvature = (left_lane_curverad + right_lane_curverad) / 2
        
    # Calculate vehicle offset
    center_offset = vehicle_position(combined_binary_warped.shape, left_fit, right_fit)
        
    # Draw lane polygon    
    result = draw_lane_polygon(undistorted, warped_img, 
                               left_line.average_fit, right_line.average_fit, 
                               Minv, lane_curvature, center_offset)

    # Assign Current fit as previous fit for next iteration   
    left_line.previous_fit = left_line.current_fit
    right_line.previous_fit = right_line.current_fit

    return result
    
    
if __name__ == '__main__':
    
    import argparse
    from moviepy.editor import VideoFileClip
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default='project_video.mp4',
                        help="specify path with image file name as PATH \
                        (without -p the default is: %(default)s)")
    args = parser.parse_args()
    print("Path to File(s) set to: {}".format(args.path))
        
    video_input = VideoFileClip(args.path)#.subclip(39, 42)
    video_output = video_input.fl_image(lambda img: pipeline(img))
    video_output.write_videofile('output_images/test_video/output_video.mp4',
                                 audio=False)
    
        
        
        
        
        
        
        