import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
"""在此必须加cv。CAP_DSHOW
不然释放资源的时候会警告"""
cap =cv.VideoCapture(0,cv.CAP_DSHOW)#加载镜头，0默认为电脑镜头

sucess=cap.isOpened()#查看摄像头是否初始化，若成功则返回True
while(sucess==True):
    """读取每一帧，read()会返回一个true和每一帧的数据，实际就是一个矩阵"""
    ret,frame=cap.read()
    if ret==True:
        """读取每一帧"""
        cv.imshow('cap',frame)
        """不加&0xFF==ord('q')的话视频会卡死"""
    if cv.waitKey(25)&0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()

