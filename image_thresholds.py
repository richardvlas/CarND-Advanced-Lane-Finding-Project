import numpy as np
import cv2


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    '''Function that applies Sobel x or y, then takes 
       an absolute value and applies a threshold.
    Parameters
    :img:    image array in RGB format
    :orient: specify gradient direction by orient = 'x' or 'y'
    :thresh: gradient min and max threshold value
    Return
    :binary_output: thresholded image
    '''
    # Convert to hls
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
        
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.abs(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    
    # Return this mask as binary_output image
    return binary_output


def hls_thresh(img, channel='s', thresh=(0,255)):
    '''Function that applies a threshold to one of hls channel.
    Parameters
    :img:     image array in RGB format
    :channel: specify one of hls channels by channel = 'h' or 'l' or 's'
    :thresh:  color channel min and max threshold value
    Return
    :binary_output: thresholded image
    '''
    # Convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Extract each color channel
    if channel == 'h':
        color_ch = hls[:,:,0]
    elif channel == 'l':
        color_ch = hls[:,:,1]
    elif channel == 's':
        color_ch = hls[:,:,2]
    binary_output = np.zeros_like(color_ch)
    binary_output[(color_ch > thresh[0]) & (color_ch <= thresh[1])] = 1
    
    return binary_output


def lab_thresh(img, channel='b', thresh=(0,255)):
    '''Function that applies a threshold to one of lab channel.
    Parameters
    :img:     image array in RGB format
    :channel: specify one of hls channels by channel = 'l' or 'a' or 'b'
    :thresh:  color channel min and max threshold value
    Return
    :binary_output: thresholded image
    '''
    # Convert to lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Extract each color channel
    if channel == 'l':
        color_ch = lab[:,:,0]
    elif channel == 'a':
        color_ch = lab[:,:,1]
    elif channel == 'b':
        color_ch = lab[:,:,2]
    binary_output = np.zeros_like(color_ch)
    binary_output[(color_ch > thresh[0]) & (color_ch <= thresh[1])] = 1
    
    return binary_output


def thresholds(img):
    '''Calculate combination of all thresholds
    Parameters
    :img:   image array in RGB format
    Return
    :combined_binary: thresholded image through all filters selected
    '''
    gradx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10,255))

    b_binary = lab_thresh(img, channel='b', thresh=(140,255))
    l_binary = lab_thresh(img, channel='l', thresh=(80,255))
    
    # Extact yellow color
    yellow_binary = np.zeros_like(gradx_binary)
    yellow_binary[(b_binary == 1) & (gradx_binary == 1)] = 1
    # Extact white color
    white_binary = np.zeros_like(gradx_binary)
    white_binary[(l_binary == 1) & (gradx_binary == 1)] = 1
    # Combined binary
    combined_binary = np.zeros_like(gradx_binary)
    combined_binary[(yellow_binary == 1) | (white_binary == 1)] = 1
        
    return combined_binary


if __name__ == '__main__':
    
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default='test_images/test5.jpg',
                        help="specify path with image file name as PATH \
                        (without -p the default is: %(default)s)")
    args = parser.parse_args()
    print("Path to File Set to: {}".format(args.path))
    
    # Read in an image from args.path
    img = cv2.imread(args.path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = thresholds(img_rgb)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    f.tight_layout()
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(result, cmap='gray')
    ax2.set_title('Thresholded Image', fontsize=15)    
    plt.show()

    