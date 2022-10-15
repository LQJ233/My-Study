import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread('C:\\Users\\ASUS\\Desktop\\333.jpg',0)
kernel=np.ones((5,5),np.uint8)
img1=cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
img2=cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
img3=cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel)
img4=cv.morphologyEx(img,cv.MORPH_BLACKHAT,kernel)
cv.imshow('show1',img1)
cv.imshow('show',img)
cv.imshow('show2',img2)
cv.imshow('show3',img3)
cv.imshow('show4',img4)
cv.waitKey(0)
cv.destroyAllWindows()
