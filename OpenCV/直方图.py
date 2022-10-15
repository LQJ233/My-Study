import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread('C:\\Users\\ASUS\\Desktop\\222.jpg')
plt.figure(1)
plt.subplot(211)
img1=cv.calcHist([img],[0],None,[255],[1,255])
plt.plot(img1)#显示直方图，好用推荐
plt.grid()#显示网格

"""绘制2d直方图"""
plt.subplot(212)#进入2行1列画板的第一个位置
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)#转换为hsv
img2=cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])#【0,1】代表直方图输入h，s两个通道，【180,256】：h为180，s为256，h取值为0到180，s取值为0到256
#x轴显示h，y轴显示s
plt.plot(img2)


plt.show()
