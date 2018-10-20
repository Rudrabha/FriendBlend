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
	for (x,y,w,h) in faces: 
		x1 = int(x-1.5*w)
		y1 = int(y-h)
		cv2.rectangle(img,(x1,y1),(x+2*w,img.shape[0]),(255,255,0),2)  
	return img


def keypoints(img):
	img_corrected = lab_contrast(img,7)
	fast = cv2.FastFeatureDetector_create()
	keypoints = fast.detect(img_corrected.copy())
	kp = []
	for i in range(len(keypoints)):
		kp.append([keypoints[i].pt[0], keypoints[i].pt[1]])
	
	return kp

