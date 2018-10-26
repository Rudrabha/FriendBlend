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
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	list1=[]
	for (x,y,w,h) in faces: 
		x1 = int(x-1.5*w)
		y1 = int(y-h)
		cv2.rectangle(img,(x1,y1),(x+2*w,img.shape[0]),(255,255,0),2)
		list1.append((x1,y1,x+2*w,img.shape[0]))
	return list1, img


def keypoints(img):
	img_corrected = lab_contrast(img,7)
	fast = cv2.FastFeatureDetector_create()
	keypoints = fast.detect(img_corrected.copy())
	kp = []
	for i in range(len(keypoints)):
		kp.append([keypoints[i].pt[0], keypoints[i].pt[1]])
	
	return kp

def keypoints_orb_matcher(img):
	orb = cv.ORB_create(nfeatures=1000)
	k = orb.detect(img)
	return k

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

