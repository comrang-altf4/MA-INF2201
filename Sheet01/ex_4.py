import cv2 as cv2
import numpy as np

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
def main():
    img=cv2.imread('bonn.png')
    sigma = -2*np.sqrt(2)
    #gaussian blur
    img_blur=cv2.GaussianBlur(img,(5,5),sigma)
    display_image("Gaussian Blur", img_blur)
    #filter2D
    gaussian_kernel = np.array([[1, 4, 6, 4, 1],