import cv2 as cv
import cv2
from lib.util import *

#some error with homography
#check
Image_1 = cv2.imread("data/1.jpg")
Image_2 = cv2.imread("data/2.jpg")

Image_1 = lab_contrast(Image_1)
Image_2 = lab_contrast(Image_2)

keypoints_1 = keypoints_orb_detector(Image_1,10000)
keypoints_2 = keypoints_orb_detector(Image_2,10000)

body_1 = detect_body(Image_1)
body_2 = detect_body(Image_2)

keypoints_valid_1 = valid_keypoints(body_1,body_2,keypoints_1)
keypoints_valid_2 = valid_keypoints(body_1,body_2,keypoints_2)

_, descriptor1  = keypoints_orb_descriptor(Image_1,keypoints_valid_1, 10000 )
_, descriptor2  = keypoints_orb_descriptor(Image_2,keypoints_valid_2, 10000 )

keypoint_matches = keypoint_bf_matcher(descriptor1, descriptor2)
source_points, destination_points = extract_matched_points(keypoint_matches, keypoints_valid_1, keypoints_valid_2)

homography_matrix = calculate_homography_matrix(source_points, destination_points)

homography_warped = warp_perspective(Image_1.copy(), Image_2.copy(), homography_matrix)

cv2.imwrite("./data/homography_output.jpg", homography_warped)

