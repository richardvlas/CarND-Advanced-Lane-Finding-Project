import numpy as np
import cv2

def image_warp(img, reversed=False, box=False):
    """Calculates perspective transform function
    Parameters
    :img:      undistored image array
    :reversed: If set True, compute the inverse to unwarp the image
    :box:      plot calibration box in source (original) and destination
               (desired or warped) coordinates
    Return
    :warped:   perspective transformed image
    """
    # Define four source coordinates (calibration box)
    src = np.array([[689,450],
                    [1038,675], 
                    [280,675],
                    [594,450]], dtype=np.float32)
    
    # Four desired or warped (dst - destination) points
    offset = 280   
    dst = np.float32([[1279-offset, 0],
                      [1279-offset, 719],
                      [offset, 719],
                      [offset, 0]])
       
    if reversed == False:
        # Compute the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)        
        warped_img = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        if box == True:
            cv2.polylines(img, np.int32([src]), True, (255,0,255),thickness=2)
            cv2.polylines(warped_img, np.int32([dst]), True, (255,0,255),thickness=4)
        # Returns warped image - uses linear interpolation
        return warped_img
    
    elif reversed == True:
        # Compute the inverse to unwarp the image by swapping the input parameters
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Returns unwarped image - uses linear interpolation
        unwarped_img = cv2.warpPerspective(img, Minv, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return unwarped_img, Minv
    
    return None


if __name__ == '__main__':
    
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default='test_images/test5.jpg',
                        help="specify path with image file name as PATH \
                        (without -p the default is: %(default)s)")
    parser.add_argument("-b", "--box", action='store_true',
                        help="plot calibration box in source (original) and destination \
                        (desired or warped) coordinates")
    args = parser.parse_args()    
    print("Path to File Set to: {}".format(args.path))
    
    # Read in an image from args.path
    img = cv2.imread(args.path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warped_img = image_warp(img_rgb, reversed=False, box=args.box)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.imshow(img_rgb)
    ax1.set_title('Source Image', fontsize=10)
    ax2.imshow(warped_img)
    ax2.set_title('Warped Image', fontsize=10)    
    plt.show()  

    
    
    
    
    
    
    
    
    
    
    