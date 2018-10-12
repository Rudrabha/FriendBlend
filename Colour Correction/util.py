import cv2 as cv

def lab_contrast(img,f):
    i_lab = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
    i_lab_split = img1_lab = cv.split(i_lab)
    clh = cv.createCLAHE(clipLimit=2.0, tileGridSize=(f,f))
    i_lab_split[0] = clh.apply(i_lab_split[0])
    i_lab = cv.merge(i_lab_split)
    op_i = cv.cvtColor(i_lab, cv.COLOR_LAB2BGR)
    return op_i

