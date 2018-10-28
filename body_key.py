import lib.util as util
import cv2
#Test
def draw_keypoints(img, kp):
	for i in range(len(kp)):
		 cv2.circle(img,(int(kp[i][0]), int(kp[i][1])), 1, (0,0,255), -1)
	return img


img = cv2.imread("data/1.jpg")
img = util.lab_contrast(img)
c1 = util.detect_body(img)
img_2 = cv2.imread("data/2.jpg")
img_2 = util.lab_contrast(img_2)
c2 = util.detect_body(img_2)

k1 = util.keypoints_orb_detector(img, 10000)
k2 = util.keypoints_orb_detector(img_2, 10000)
k_u1 = util.valid_keypoints(c1, c2, k1)
k_u2 = util.valid_keypoints(c1, c2, k2)

_, des_u1 = util.keypoints_orb_descriptor(img, k_u1, 10000)
_, des_u2 = util.keypoints_orb_descriptor(img_2, k_u2, 10000)
dmatches = util.keypoint_bf_matcher(des_u1, des_u2)
src_pts, dst_pts = util.extract_matched_points(dmatches, k_u1, k_u2)
h = util.calculate_homography_matrix(src_pts, dst_pts)
im_out = util.warp_perspective(img.copy(), img_2.copy(), h)
cv2.imwrite("homography.jpg", im_out)

