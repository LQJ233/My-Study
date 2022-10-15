import cv2 as cv
img=cv.imread('C:\\Users\\ASUS\\Desktop\\222.jpg',0)
ret,thresh=cv.threshold(img,127,255,0)
counts,aaa=cv.findContours(thresh,1,2)
cnt=counts[0]
m=cv.moments(cnt)
print(m)