import cv2 as cv
cap=cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,488)
classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

    configPath=