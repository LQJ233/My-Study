import cv2 as cv
import matplotlib.pyplot as plt
"""读取视频"""
cap=cv.VideoCapture(0)
"""在每一帧数据中进行人脸识别"""
face_casecade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        """实例化opencv人脸识别分类器"""
        face_cas=cv.CascadeClassifier("haarcascade_fromtalface_default.xml")
        face_cas.load("haarcascade_fromtalface_default.xml")
        faceRects=face_cas.detectMultiScale(gray,1.3,2)
        for faceRects in faceRects:
            x,y,w,h=faceRects
            """框出人脸"""
            cv.rectangle(frame,(x,y),(x+h,h+w),(0,225,0),3)
        cv.imshow("frame",frame)
        if cv.waitKey(1)& 0xFF==ord('q'):
            break
"""释放资源"""
cap.release()
cv.destroyAllWindows()
