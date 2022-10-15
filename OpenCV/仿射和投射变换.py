import cv2 as cv
import numpy as np

img=cv.imread('C:\\Users\\ASUS\\Desktop\\222.png',1)
pts1=np.float64([[123,123],[234,234],[345,345]])#获取三个点
pts2=np.float64([[789,789],[567,567],[555,555]])#变化后三个点的位置，整幅图跟着变化

x,y=img.shape[:2]
m=cv.getAffineTransform(pts1,pts2)#affine意为仿射
change_img=cv.warpAffine(img,m,(x,y))
cv.waitKey(0)
cv.destroyAllWindows()