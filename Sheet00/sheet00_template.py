import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png' 
    
    img = cv.imread(img_path)
    # 2a: read and display the image 
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    img_gray_05 = img_gray*0.5
    img_cpy = img.copy()
    for i in range(3):
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                img_cpy[j,k,i] = max(img_cpy[j,k,i] - img_gray_05[j,k],0)
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    img_cpy = img.copy()
    img_cpy = cv.cvtColor(img_cpy, cv.COLOR_BGR2RGB)
    img_gray_3d = cv.merge([img_gray_05, img_gray_05, img_gray_05]).astype(np.uint8)
    img_cpy = cv.subtract(img_cpy, img_gray_3d)
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    # 2e: Extract the center patch and place randomly in the image
    img_cpy = img.copy()
    img_patch = img_cpy[img_cpy.shape[0]//2-8:img_cpy.shape[0]//2+8, img_cpy.shape[1]//2-8:img_cpy.shape[1]//2+8,:]

    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement
    rand_coord = [randint(0, img_gray.shape[0]-16), randint(0, img_gray.shape[1]-16)]
    img_cpy[rand_coord[0]:rand_coord[0]+16, rand_coord[1]:rand_coord[1]+16] = img_patch
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)  
    # 2f: Draw random rectangles and ellipses
    img_cpy = img.copy()
    for i in range(5):
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        thickness = randint(1, 5)
        cv.rectangle(img_cpy, (randint(0, img_gray.shape[1]), randint(0, img_gray.shape[0])), (randint(0, img_gray.shape[1]), randint(0, img_gray.shape[0])), color, thickness)
        cv.ellipse(img_cpy, (randint(0, img_gray.shape[1]-16), randint(0, img_gray.shape[0]-16)), (randint(0, 100), randint(0, 100)), randint(0, 360), 0, 360, color, thickness)
    display_image('2 - f - Rectangles and Ellipses', img_cpy)
       
    # # destroy all windows
    cv.destroyAllWindows()
