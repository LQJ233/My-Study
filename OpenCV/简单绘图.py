import cv2
import numpy as np
import cv2 as cv
def imshow(show,img):#展示图像的函数，注意形参和实参
    cv.imshow("show", img)
    cv.waitKey(0)  # k必须大写
    cv.destroyAllWindows()

img1=cv.imread("C:\\Users\\ASUS\\Desktop\\111.png",1)
black_img=np.zeros((255,255,3),np.uint8)#***=np.zeros（（），np.uint8）暂时记为固定用法，用于制备全黑图像
#img1=cv.cvtColor(img,cv2.COLOR_BGR2HSV)#图像转换颜色通道
#print(img1.shape)#显示图像长宽通道数
#print(img1[0,0])#显示图像RGB的数字，若为黑白照片则只有一个数字
cv.line(img1,(111,222),(1000,1000),(226,134,51),10)#(图像，起始坐标，结束坐标，颜色，线宽，线形状)
cv.rectangle(img1,(222,333),(444,999),(123,231,23),1,cv.LINE_AA)#cv。line_AA为锯齿线
a=np.array([[123,234],[774,2],[567,678],[789,899]],np.int64)#数组用中括号，np.array（）和np.int64为了提高效率
cv.polylines(img1,[a],True,(225,225,225),1,cv.LINE_AA)#数组用中括号括起来
#cv.putText(img1,'略略略'(555,999), font,10,(255,0,0),5,cv.LINE_AA)
img=black_img
imshow("show",img)#调用函数