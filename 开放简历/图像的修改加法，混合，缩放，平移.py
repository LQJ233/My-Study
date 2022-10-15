import cv2 as cv
import numpy as np
import cv2
def imshow(show,img):
    cv.imshow("show",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
img1=cv.imread('C:\\Users\\ASUS\\Desktop\\111.png',1)
img2=cv.imread('C:\\Users\\ASUS\\Desktop\\222.jpg',1)#(-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'为没找到照片，路径不对
x,y=img1.shape[:2]#取图像的长宽坐标，死记
#img=cv2.resize(img1,(2*x,2*y))#放大图像，绝对尺寸，只能用整数，不能用浮点，所以绝对尺寸只能放大
#img=cv.resize(img1,None,fx=2,fy=2)#相对尺寸，放大缩小尺寸都行，插入方法不会，，，迷
M=np.float64([[1,0,50],[0,1,50]])#float64为固定搭配，暂时死记
img=cv.warpAffine(img1,M,(x,y))#这里数组不加方括号，需要先获取图像的长宽
#print(cv.add(img1+img2))
#imshow('show3',img3)
#imshow('show1',img1)
#imshow("show2",img2)
imshow('show1',img)
