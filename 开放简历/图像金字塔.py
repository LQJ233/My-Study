import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread('C:\\Users\\ASUS\\Desktop\\222.jpg',1)
#img1=cv.pyrUp(img)向上采样，图片更大，更清晰
img1=cv.pyrDown(img)#图像更小，
cv.imshow('show',img1)
cv.waitKey(0)
cv.destroyAllWindows()
