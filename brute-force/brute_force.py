import numpy as np
import cv2
from matplotlib import pyplot as plt
# Initiate SURF detector
surf = cv2.xfeatures2d.SURF_create()
img_query = cv2.imread('image.orig/0.jpg',0)          # queryImage
kp2, des2 = surf.detectAndCompute(img_query,None)

for x in range(0,999,1):
    if (x % 100) !=0:
        path_img ="image.orig/" + str(x) +".jpg"
        print (path_img)
        path_kp = "keypoint/" + str(x) + ".txt"
        path_des = "des/" + str(x) + ".txt"
        img = cv2.imread(path_img, 0)  # trainImage
        # find the keypoints and descriptors with SURF
        kp, des = surf.detectAndCompute(img, None)
        np.savetxt(path_kp, kp, delimiter=" ", fmt="%s")
        np.savetxt(path_des, des, delimiter=" ", fmt="%s")

