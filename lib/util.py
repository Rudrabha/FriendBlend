import cv2,sys
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
    var = 1.5
    iter=0
    face_cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, var, 5)
    while(len(faces)==0):
        iter += 1
        var = var - 0.1
        faces = face_cascade.detectMultiScale(gray, var, 5)
        if iter>=5:
            break
    dist=0
    for (x,y,w,h) in faces:
        x1 = int(x-int(1.5*w))
        if x1<0:
            x1=0
        y1 = int(y-h)
        if y1<0:
            y1=0
        x2 = x+int(2.2*w)
        if x2>img.shape[1]:
            x2=img.shape[1]
        y2 = img.shape[0]
        area = ((x2-x1) * (y2-y1))
        if dist < area:
            dist = area
            list1 = [(x1,y1,x2,y2,w,h)]
    img = cv2.rectangle(img,(list1[0][0],list1[0][1]),(list1[0][2],list1[0][3]),(255,0,0),2)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return list1,img


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

def keypoint_bf_matcher(des1, des2, n=40):
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	min_dist = matches[0].distance	
	if (len(matches) < 500):
		n = 20
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
	top_left_x1,top_left_y1,bot_right_x1,bot_right_y1,_,_=body1[0]
	top_left_x2,top_left_y2,bot_right_x2,bot_right_y2,_,_=body2[0]
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
    new_points[new_points <0] = 0
    new_points = new_points.reshape((new_points.shape[0], new_points.shape[2]))
    return new_points

def sort_order(img1,img2,body1,body2):
    col_start = body1[0][2]
    col_end= body2[0][0]
    if col_start>col_end:
        img_temp = img1.copy()
        img1=img2.copy()
        img2 = img_temp.copy()
        body_temp =body1.copy()
        body1 = body2.copy()
        body2 = body_temp.copy()
    return img1,img2,body1,body2

def blend_or_cut(body1,body2, n=500):
    if body2[0][2]-body1[0][2]<n:
        print("Bodies close, Implementing GrabCut")
        return "grabcut"
    else:
        print("Bodies far enough, Implementing Alpha Blending")
        return "blend"

def grabcut(img1,img2,body1,body2):

    a1,b1,c1,d1,w1,h1 = body1[0]
    a2,b2,c2,d2,w2,h2 = body2[0]
    
    mask = np.zeros(img1.shape[:2],np.uint8)
    backgroundModel = np.zeros((1,65),np.float64)
    foregroundModel = np.zeros((1,65),np.float64)
    if w1*h1 > w2*h2:
        foreground = img1.copy()
        background = img2.copy()
        rect = (a1,b1,c1,d1)
        for i in range(a1,c1):
            for j in range(b1,d1):
                mask[j,i] = cv2.GC_PR_FGD
        for i in range(a1+w1-5,c1-w1+5):
            for j in range(b1+h1-5,b1+int(1*h1)+5):
                mask[j,i] = cv2.GC_FGD
        cv2.grabCut(foreground, mask,rect,backgroundModel,foregroundModel,1,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = foreground*mask2[:,:,np.newaxis]

        return img,background
    else:
        foreground = img2.copy()
        background = img1.copy()
        rect = (a2,b2,c2,d2)
        for i in range(a2,c2):
            for j in range(b2,d2):
                mask[j,i] = cv2.GC_PR_FGD
        for i in range(a2+w2-5,c2-w2+5):
            for j in range(b2+h2-5,b2+int(1*h2)+5):
                mask[j,i] = cv2.GC_FGD
        cv2.grabCut(foreground, mask,rect,backgroundModel,foregroundModel,1,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = foreground*mask2[:,:,np.newaxis]

        return img,background

def alpha_blend(img1, img2, body1, body2):
    op = np.zeros(img1.shape)
    col_start = body1[0][2]
    col_end= body2[0][0]
    if col_start>col_end:
        img_temp = img1.copy()
        img1=img2.copy()
        img2 = img_temp.copy()
        body_temp =body1.copy()
        body1 = body2.copy()
        body2 = body_temp.copy()
        op = np.zeros(img1.shape)
        col_start = body1[0][2]
        col_end = body2[0][0]
    
    step_size = 1/(col_end-col_start)
    for x in range(col_start,col_end):
        step_count = x-col_start
        op[:,x,0] = ((1-(step_count*step_size))*img1[:,x,0])+((step_count*step_size)*img2[:,x,0])
        op[:,x,1] = ((1-(step_count*step_size))*img1[:,x,1])+((step_count*step_size)*img2[:,x,1])
        op[:,x,2] = ((1-(step_count*step_size))*img1[:,x,2])+((step_count*step_size)*img2[:,x,2])
    op[:,0:col_start,:] = img1[:,0:col_start,:]
    op[:,col_end:,:] = img2[:,col_end:,:]
    return op

def crop_image(img, H):
	rows, cols, channels = img.shape
	pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]).reshape(-1,
		                                                           1, 2)
	warp_points = cv2.perspectiveTransform(pts, H)
	top_row = int(max(max(warp_points[0][0][1], warp_points[1][0][1]), 0))
	bottom_row = int(min(min(warp_points[2][0][1], warp_points[3][0][1]), rows))

	return img[top_row:bottom_row, 0:cols]

def blend_cropped_image(background_img, input_img):
    height, width, channels = input_img.shape
    cropped_img_binary = np.zeros((height, width, 1), np.uint8)
    cropped_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    cv2.threshold(cropped_img_gray, 0, 255, cv2.THRESH_BINARY, cropped_img_binary)
    concat_image = np.concatenate((cropped_img_gray, cropped_img_binary[:,:,0]),axis=1)
    merged_img = np.uint8((255 - cropped_img_binary) / 255) * background_img + input_img
    element_sizes = [(5, 0)]
    outer_mask = cropped_img_binary
    for erosion_size, fore_coeff in element_sizes:
        element_size = (erosion_size, erosion_size)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, element_size)
        inner_mask = np.zeros((height, width, 1), np.uint8)
        cv2.erode(outer_mask, element, dst=inner_mask)
        mask = outer_mask - inner_mask
        blended_section = np.uint8(fore_coeff * mask / 255 * input_img +(1 - fore_coeff) * mask / 255 * background_img)
        inverse_mask = 255 - mask
        inverse_mask = inverse_mask / 255  # convert to 1s and 0s
        merged_img = inverse_mask * merged_img + blended_section
        outer_mask = inner_mask
    rows, cols, channels = inner_mask.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if inner_mask[i, j] == np.uint8(255):
                bottom_row = i
                break
    merged_img = merged_img[0:bottom_row, 0:cols]
    return merged_img


