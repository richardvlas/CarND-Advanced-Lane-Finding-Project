import numpy as np
import cv2

def cam_calibration(path):
    """Function that takes images, calculates object points, image points
       and performs camera calibration
    Parameters
    :path:  Pathname pattern
    Return
    :mtx:   camera matrix
    :dist:  distortion coefficients
    """
    # Array to store object points and image points from all images 
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... (8,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Iterate through the image list (path) and search for chessboard corners
    for fname in path:
        # read in each image
        img = cv2.imread(fname)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        
    # Calculate camera calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img.shape[1::-1],
                                                       None, None)
    # Return camera matrix and distortion coefficients
    return mtx, dist

def image_undistort(img, mtx, dist):
    """Performs image distortion correction and returns the 
       undistorted image
    Parameters
    :img:    image array
    :mtx:    camera matrix
    :dist:   distortion coefficients
    Returns
    :undist: undistorted image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist


if __name__ == '__main__':
    
    import argparse
    import glob
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default='./camera_cal/calibration*.jpg',
                        help="specify path to calibration image files \
                        (without -p the default is: %(default)s)")
    args = parser.parse_args()    
    print("Path to calibration files set to: {}".format(args.path))
    
    # Read in and make a list of calibration images
    path_calibration_images = glob.glob(args.path)
    # Camera calibration: returns the camera matrix and distortion coefficients
    mtx, dist = cam_calibration(path_calibration_images)

    # Read in an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    undistorted = image_undistort(img, mtx, dist)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=15)
    plt.show()
    
    