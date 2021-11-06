import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite('sift_keypoints.jpg', img)

# sift = cv.SIFT_create()
# kp = sift.detect(gray, None)
# img = cv.drawKeypoints(gray, kp, img)
# imgplt = plt.imshow(img)
# plt.show()