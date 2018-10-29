import cv2 
import cv2 as cv
import numpy as np

def lab_contrast(img, f=7):
	i_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
	i_lab_split = img1_lab = cv.split(i_lab)
	clh = cv.createCLAHE(clipLimit=2.0, tileGridSize=(f,f))
	i_lab_split[0] = clh.apply(i_lab_split[0])
	i_lab = cv.merge(i_lab_split)
	op_img = cv.cvtColor(i_lab, cv.COLOR_LAB2BGR)
	return op_img

def detect_body(img):
	face_cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml') 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	list1=[]
	for (x,y,w,h) in faces: 
		x1 = int(x-1.5*w)
		y1 = int(y-h)
		list1.append((x1,y1,x+2*w,img.shape[0]))
	return list1


def keypoints(img):
	img_corrected = lab_contrast(img,7)
	fast = cv2.FastFeatureDetector_create()
	keypoints = fast.detect(img_corrected.copy())
	kp = []
	for i in range(len(keypoints)):
		kp.append([keypoints[i].pt[0], keypoints[i].pt[1]])
	
	return kp

def keypoints_orb_detector(img, n=1000):
	orb = cv.ORB_create(nfeatures=n)
	k = orb.detect(img)
	return k

def keypoints_orb_descriptor(img, kp, n=1000):
	orb = cv.ORB_create(nfeatures=n)
	kp, des = orb.compute(img, kp)
	return kp, des

def keypoint_bf_matcher(des1, des2, n=50):
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	return matches[0:n]

def extract_matched_points(dmatches, kpts1, kpts2):
	src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
	dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
	return src_pts, dst_pts

def calculate_homography_matrix(pts_src, pts_dst):
	h, status = cv2.findHomography(pts_src, pts_dst)
	return h

def warp_perspective(img_src, h):
	im_out = cv2.warpPerspective(img_src, h, (img_src.shape[1],img_src.shape[0]))
	return im_out

def valid_keypoints(body1,body2,keypoints):
	op_keypoints = keypoints.copy()
	top_left_x1,top_left_y1,bot_right_x1,bot_right_y1=body1[0]
	top_left_x2,top_left_y2,bot_right_x2,bot_right_y2=body2[0]
	for i in range(len(keypoints)):
		point = keypoints[i].pt
		if (((point[0]<top_left_x1 or point[0]>bot_right_x1) or (point[1]<top_left_y1 or point[1]>bot_right_y1)) and ((point[0]<top_left_x2 or point[0]>bot_right_x2) or (point[1]<top_left_y2 or point[1]>bot_right_y2))):
			continue
		else:
			op_keypoints.remove(keypoints[i])
	return op_keypoints

def L2_distance(pt1, pt2):
	return np.linalg.norm(pt1-pt2)

def list2pt(keypoints):
	kp = []
	for i in range(len(keypoints)):
		kp.append([keypoints[i].pt[0], keypoints[i].pt[1]])
	kp = np.asarray(kp)
	return kp

def draw_keypoints(img, kp):
	for i in range(len(kp)):
		 cv2.circle(img,(int(kp[i][0]), int(kp[i][1])), 2, (0,0,255), -1)
	return img

def transform_points(pt1, homography_matrix):
	new_points = cv2.perspectiveTransform(pt1, homography_matrix)
	new_points = new_points.reshape((new_points.shape[0], new_points.shape[2]))
	return new_points

def alpha_blend(img1, img2, body1, body2):
    op = np.zeros(img1.shape)
    _,_,col_start,_ = body1[0]
    col_end,_,_,_ = body2[0]
#    print(body1,body2)
#    print(col_start,col_end)

    step_size = 1/(col_end-col_start)
    for x in range(col_start,col_end+1):
        step_count = x-col_start
        op[:,x,0] = ((1-(step_count*step_size))*img1[:,x,0])+((step_count*step_size)*img2[:,x,0])
        op[:,x,1] = ((1-(step_count*step_size))*img1[:,x,0])+((step_count*step_size)*img2[:,x,1])
        op[:,x,2] = ((1-(step_count*step_size))*img1[:,x,0])+((step_count*step_size)*img2[:,x,2])
    op[:,0:col_start,:] = img1[:,0:col_start,:]
    op[:,col_end:,:] = img2[:,col_end:,:]
    return op

