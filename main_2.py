import cv2 as cv
import cv2
from lib.util import *

#some error with homography
#check
Image_1 = cv2.imread("data/1.jpg")
Image_2 = cv2.imread("data/2.jpg")
homography_warped_1 = homography_warp(Image_1, Image_2)
op_image = 0.5*Image_2 + 0.5*homography_warped_1
cv2.imwrite("./data/alpha_blend2.jpg", op_image)

