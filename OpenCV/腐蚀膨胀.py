import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread('C:\\Users\\ASUS\\Desktop\\333.jpg',1)
resize_img=cv.resize(img,None,fx=0.5,fy=0.5)

kernel=np.ones((5,5),np.uint8)
#erode_img=cv.erode(resize_img,kernel,1)#腐蚀
dilate_img=cv.dilate(resize_img,kernel)#膨胀
cv.imshow("show",dilate_img)
cv.waitKey(0)
cv.destroyAllWindows()