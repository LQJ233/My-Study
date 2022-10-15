import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
img=cv.imread("C:\\Users\\ASUS\\Desktop\\111.png",0)
imge,contours=cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
img_contours=cv.drawContours(img,contours,(0,255,0),3)
cv.imshow('img',img_contours)
cv.waitKey(0)&0xFF
cv.destroyAllWindows()