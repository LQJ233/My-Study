import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def nothing (x):
    pass

"""创建一个黑色的窗口"""
img=np.zeros((300,512,3),np.uint8)
"""给这个窗口命个名"""
cv.namedWindow(('imge'))

"""创建几个轨迹栏"""
cv.createTrackbar('B','imge',int(0),int(255),nothing)

while (1):
    cv.imshow('imge',img)
    k=cv.waitKey(0)&0xFF
    if k==27:
        break
    b=cv.getTrackbarPos('B','imge')
   # s=cv.getTrackbarPos()
    #if s==0:
     #   img[:]=0
    #else:
      #  img=[b]
cv.destroyAllWindows()

