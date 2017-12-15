import numpy as np
import cv2
from matplotlib import pyplot as plt
# Initiate SURF detector
surf = cv2.xfeatures2d.SURF_create()
img_query = cv2.imread('image.orig/0.jpg',0)          # queryImage
kp_quey, des_query = surf.detectAndCompute(img_query,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
des_all = None
flann = cv2.FlannBasedMatcher(index_params,search_params)
for x in range(0,9,1):
    # if (x % 100) !=0:
        path_img ="image.orig/" + str(x) +".jpg"
        print (x)
        path_kp = "keypoint/" + str(x) + ".txt"
        path_des = "des/" + str(x) + ".txt"
        img = cv2.imread(path_img, 0)  # trainImage
        # find the keypoints and descriptors with SURF
        kp, des = surf.detectAndCompute(img, None)

        if des_all is None:
            des_all = des
        else:
            des_all = np.concatenate((des_all, des))

flann = cv2.flann.Index()
print ("Training...")
flann.build(des_all, index_params)
print ("Matching...")
indexes, matches = flann.knnSearch(des_query, 10)

# print (matches)
# matches = flann.knnMatch(des, des_query, k=2)
        # matchesMask = [[0, 0] for i in range(len(matches))]
        # # ratio test as per Lowe's paper
        # cnt = 0
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.7 * n.distance:
        #         cnt = cnt + 1
        #         matchesMask[i] = [1, 0]
        # if (cnt / (max(len(des), len(des_query)))) > 0.01:
        #     print (path_img)
        #     print ("match")

        # np.savetxt(path_kp, kp, delimiter=" ", fmt="%s")
        # np.savetxt(path_des, des, delimiter=" ", fmt="%s")