import cv2 as cv
import cv2,sys
from lib.util import *

#some error with homography
#check
n_keypoints = 10000
Image_1 = cv2.imread("dataset/im10_1.jpeg")
Image_2 = cv2.imread("dataset/im10_2.jpeg")

x,y,_ = Image_1.shape
Image_2 = cv2.resize(Image_2,(y,x))

Image_1 = lab_contrast(Image_1)
Image_2 = lab_contrast(Image_2)

body_1, i_b1 = detect_body(Image_1.copy())
body_2, i_b2 = detect_body(Image_2.copy())

if (len(body_1) == 0 or len(body_2) == 0):
    print("Exitting the process as **Face not detected in one/both Images**")
    sys.exit()

Image_1,Image_2,body_1,body_2 = sort_order(Image_1,Image_2,body_1,body_2)


keypoints_1 = keypoints_orb_detector(Image_1,n_keypoints)
keypoints_2 = keypoints_orb_detector(Image_2,n_keypoints)

if (len(body_1) == 0 or len(body_2) == 0):
    print("Exitting the process as **Face not detected in one/both Images**")
    sys.exit()

keypoints_valid_1 = valid_keypoints(body_1,body_2,keypoints_1)
keypoints_valid_2 = valid_keypoints(body_1,body_2,keypoints_2)

_, descriptor1  = keypoints_orb_descriptor(Image_1,keypoints_valid_1, n_keypoints)
_, descriptor2  = keypoints_orb_descriptor(Image_2,keypoints_valid_2, n_keypoints)

keypoint_matches = keypoint_bf_matcher(descriptor1, descriptor2)

source_points, destination_points = extract_matched_points(keypoint_matches, keypoints_valid_1, keypoints_valid_2)

homography_matrix = calculate_homography_matrix(source_points, destination_points)
#
homography_warped_1 = warp_perspective(Image_1.copy(), homography_matrix)

top_left_x1,top_left_y1,bot_right_x1,bot_right_y1,_,_=body_1[0]
pt1 = np.float32([[[top_left_x1, top_left_y1]],[[bot_right_x1, top_left_y1]],[[top_left_x1, bot_right_y1]] ,[[bot_right_x1,bot_right_y1]]])

new_points = transform_points(pt1, homography_matrix)
new_points[new_points<0] = 0
new_points= new_points.astype(int)
a,b = new_points[0]
c,d = new_points[-1]
body_1_homographed = [(a,b,c,d)]
op_image = alpha_blend(homography_warped_1,Image_2,body_1_homographed,body_2)
cv2.imwrite("trial_outputs/im10_op.jpg", op_image)

