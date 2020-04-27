import numpy as np
import matplotlib.pyplot as plt
import cv2


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.current_fit = np.array([0, 0, 0])
        # polynomial coefficients for the previous fit
        self.previous_fit = np.array([0, 0, 0])
        # polynomial coefficients for previous n iterations
        self.n_fits = []
        # average polynomial coefficients for previous n iterations
        self.average_fit = np.array([0, 0, 0])
    
    
def validate(img_shape, left_line, right_line):
    # define how to validate that lane lines are detected properly
    # Calculate lane width
    left_fitx, right_fitx, ploty = fit_x_values(img_shape, 
                                                left_line.current_fit,
                                                right_line.current_fit)

    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    lane_width_top = (right_fitx[0] - left_fitx[0])*xm_per_pix
    lane_width_bottom = (right_fitx[-1] - left_fitx[-1])*xm_per_pix
    diff_top_bottom = lane_width_top - lane_width_bottom
    
    san_check = True
    # Check that the curvature sing is the same
    if np.sign(left_line.current_fit[0]) != np.sign(right_line.current_fit[0]):
        print('Oposite Curvature DETECTED!')
        left_line.current_fit = left_line.previous_fit
        right_line.current_fit = right_line.previous_fit
        san_check = False
    
    # Check that the curvature is similar between left and right lane lines
    if (abs(left_line.current_fit[0])/abs(right_line.current_fit[0]) < 1/3.0) or (abs(left_line.current_fit[0])/abs(right_line.current_fit[0]) > 3.0):
        print('Different Radius between left and right Line!')
        left_line.current_fit = left_line.previous_fit
        right_line.current_fit = right_line.previous_fit
        san_check = False
    
    # Check that the lane lines don't cross each other
    if lane_width_top < 0 or lane_width_bottom < 0:
        left_line.current_fit = left_line.previous_fit
        right_line.current_fit = right_line.previous_fit
        san_check = False
    
    # Assign True or False depending on Sanity Check san_check
    left_line.detected = san_check
    right_line.detected = san_check
    
    return None


def fit_poly(leftx, lefty, rightx, righty):
    # Fit a second order polynomial with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit
    

def fit_x_values(img_shape, left_fit, right_fit):
    # Generate y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calculate x values for both polynomials using ploty, left_fit and right_fit
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Return x values of polynomial
    return left_fitx, right_fitx, ploty


def sliding_window(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_image = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint    
    
    ## HYPERPARAMETERS ##
    # Choose the number of sliding windows
    nwindows = 20 
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices 
    left_lane_inds  = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (for right and left lane)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # Find the four x boundaries of the window
        win_xleft_low   = leftx_current - margin
        win_xleft_high  = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_image, (win_xleft_low, win_y_low),
                                 (win_xleft_high, win_y_high),(0,255,0), 2)
        cv2.rectangle(out_image, (win_xright_low, win_y_low),
                                 (win_xright_high, win_y_high),(0,255,0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If pixels found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each lane pixel using `np.polyfit`
    left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)
            
    ## Visualization ##
    # Colors in the left(red) and right(blue) lane regions
    out_image[lefty, leftx] = [255, 0, 0]
    out_image[righty, rightx] = [0, 0, 255]
        
    return left_fit, right_fit, out_image


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100
    
    # Grab activated pixels
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values within the +/- margin of our polynomial function 
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                                   left_fit[2] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                                    right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each lane pixel
    left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)
    
    # Calculate x values of polynomials and return ploty as well
    left_fitx, right_fitx, ploty = fit_x_values(binary_warped.shape, left_fit, right_fit)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_image = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    ## End visualization steps ##
        
    return left_fit, right_fit, out_image


def avg_fit(line, n=3):
    average_pol_coeff = np.zeros(n)    
    # First add latest fit into the list 
    line.n_fits.append(line.current_fit)
    if len(line.n_fits) > n:
        line.n_fits.pop(0)
    n_fits_array = np.vstack(line.n_fits)
    average_pol_coeff = n_fits_array.mean(axis=0)

    return average_pol_coeff


def draw_lane_lines(img, left_fit, right_fit):
    # Calculate x values of polynomials and return ploty as well
    left_fitx, right_fitx, ploty = fit_x_values(img.shape, left_fit, right_fit)
    
    return left_fitx, right_fitx, ploty
    

def draw_lane_polygon(undist_img, warped_img, left_fit, right_fit, Minv, 
                      lane_curvature, center_offset):
    
    # Calculate x values of polynomials and return ploty as well
    left_fitx, right_fitx, ploty = fit_x_values(warped_img.shape, left_fit, right_fit)
    
    # Create an image to draw on and an image to show the selection window
    lane_img = np.zeros_like(warped_img)
       
    # Generate a polygon to illustrate the lane area
    # will recast the x and y points into usable format for cv2.fillPoly()
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    lane_line_pts = np.hstack((left_line_pts, right_line_pts))
        
    # Draw the lane onto the warped blank image
    cv2.fillPoly(lane_img, np.int_([lane_line_pts]), (0, 255, 0))
    
    unwarped_lane_img = cv2.warpPerspective(lane_img, Minv, (undist_img.shape[1],
                                                             undist_img.shape[0]))

    # Overlay original images with lane polygon on top of it
    result = cv2.addWeighted(undist_img, 1, unwarped_lane_img, 0.3, 0)
    
    # Add Curvature and center offset information
    current_curvature = "Lane Curvature: {:5.0f} [m]".format(lane_curvature)
    current_offset    = "Center Offset:  {:5.2f} [m]".format(center_offset) 
    
    # Draws text on image
    cv2.putText(result, current_curvature, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255, 255))
    cv2.putText(result, current_offset, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255, 255))
    
    return result

    
def measure_curvature_real(img_shape, fit):
    '''Calculates the curvature of polynomial function in meters.
    '''
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # vector representing y values to cover y-range as image
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    
    # Define y-value where the radius of curvature will be calculated
    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of radius of curvature
    curverad = (1 + (2*fit[0]*y_eval*ym_per_pix + fit[1])**2)**(3/2) / np.abs(2*fit[0])
            
    return curverad


def vehicle_position(img_shape, left_fit, right_fit):
    '''Calculates vehicle position with respect to center of lane in meters.
       This is equivalent to offset from the center of the lane.
       Positive value - vehicle is to the right of the lane center
       Negative value - vehicle is to the left of the lane center
    '''
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    car_position = img_shape[1] // 2
    
    left_fitx, right_fitx, ploty = fit_x_values(img_shape, left_fit, right_fit)

    lane_center = (right_fitx[-1] + left_fitx[-1]) / 2
    vehicle_offset = (car_position - lane_center)*xm_per_pix
        
    return vehicle_offset

