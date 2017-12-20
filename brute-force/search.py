import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
img_search = cv2.imread('/home/thao-nt/Desktop/MMDB/MMDB/bag-of-words/dataset/train2/food/900.jpg',0)          # queryImage
# img2 = cv2.imread('/home/thao-nt/Desktop/MMDB/MMDB/bag-of-words/dataset/train2/beach/107.jpg',0) # trainImage

# Initiate SIFT detector
surf = cv2.xfeatures2d.SURF_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img_search,None)
# kp2, des2 = surf.detectAndCompute(img2,None)
# np.savetxt("des1.txt", des1, delimiter=" ", fmt="%s")
# np.savetxt("des2.txt", des2, delimiter=" ", fmt="%s")
# diffs=np.intersect1d(des1,des2)
# print (len(diffs))
# np.savetxt("diffs1.txt", diffs, delimiter=" ", fmt="%s")
train_path = "des_temp/"
training_names = os.listdir(train_path)
cnt_arr=[]
url_arr=[]
for training_name in training_names:
    # print (training_name)
    # des2=np.loadtxt("des/"+training_name, delimiter=" ",usecols=(0, 2), unpack=True)
    des2 = np.load('des_arr/'+training_name)
    # print (des2)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # print (len(des1))
    # print (len(des2))
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    cnt=0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            cnt = cnt+1
            matchesMask[i]=[1,0]
    # print (cnt)
    # print (training_name)
    cnt_arr.append(cnt)
    url_arr.append(training_name)
    Z = [url for _, url in sorted(zip(cnt_arr, url_arr),reverse=True)]
    # cnt_arr.sort(reverse=True);
print (len(Z))
    # if (cnt/(max(len(des1),len(des2)))) >0.5:
    #     print ("match")
    # else:
    #     print ("not match")
    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = 0)
    #
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    #
    # plt.imshow(img3,),plt.show()