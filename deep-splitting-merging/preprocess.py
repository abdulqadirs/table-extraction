import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def remove_lines(orig_image, binary_img):
    src = orig_image
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
        
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)
   
    cols = horizontal.shape[1]
    horizontal_size = cols // 30

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    #verticalsize = rows // 40
    verticalsize = 35

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)



    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (39,3))
    horz_dilated = cv2.dilate(horizontal, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
    vert_dilated = cv2.dilate(vertical, kernel, iterations=1)
    
    combined = cv2.add(horz_dilated,vert_dilated)
    
    
    combined = cv2.add(binary_img,combined)
    
    return combined

def binarize_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,17)
    return img

def preproces_img(img):

    bin_img = binarize_img(img)
    img = remove_lines(img, bin_img)
    img = Image.fromarray(img)

    transform = transforms.ToTensor()
    img = transform(img)

    return img
