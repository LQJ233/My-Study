import cv2 as cv
import numpy as np
import cv2
img=cv.imread('C:\\Users\\ASUS\\Desktop\\111.png',1)
x,y=img.shape[:2]
print (x,y)
print(type(x))#2.查了一下x的数据类型发现本来就是int
m=cv.getRotationMatrix2D((x/2,y/2),130,1)

ratation_img=cv.warpAffine(img,m,(x,y))
cv.circle(ratation_img,(int(x/2),int(y/2)),5,(0,0,255))#1.不知道为什么这里画圆的中心坐标必须要强制转化
cv.imshow('show',ratation_img)
cv.waitKey(0)
cv.destroyAllWindows()
