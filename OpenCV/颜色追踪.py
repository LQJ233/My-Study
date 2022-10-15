"""import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
cap=cv.VideoCapyure(0)
while (cap.isOpened==True):
    ret,frame=cap.read()
    HSV_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    red_frame=cv.threshold(HSV_frame,,255,cv.THRESH.BINARY)
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
cap=cv.VideoCapture(0)
while(1):
    """获取每一帧"""
    ret, frame = cap.read()
    """转换到HSV"""

    HSV_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    """设定蓝色阈值"""

    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])
    """根据阈值创建掩膜"""

    mask=cv.inRange(HSV_frame,lower_blue,upper_blue)
    """对原图像进行位运算"""
    res=cv.bitwise_and(frame,frame,mask=mask)

    """显示图像"""
    cv.imshow("frame",frame)
    cv.imshow("mask",mask)
    cv.imshow("res",res)

    cv.waitKey(25)&0xFF


cv.destroyAllWindows()