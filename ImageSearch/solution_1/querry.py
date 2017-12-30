import argparse as ap
import cv2
import os
import csv
import numpy as np

feature_paths = "extract-feature"
img_querry_path = "../data-test/326.jpg"


surf = cv2.xfeatures2d.SURF_create()
img_querry = cv2.imread(img_querry_path,0) 
kp, des = surf.detectAndCompute(img_querry,None)
bf = cv2.BFMatcher()

result = []

for feature_path in os.listdir(feature_paths):
    temp_des = []                                       #luu cac des duoc lay tu file 
    print("Xu Ly Anh: ", feature_path)


    with open('extract-feature/' + feature_path, 'rb') as csvfile:
        matrixreader = csv.reader(csvfile, delimiter=' ')
        image_compare = "".join(next(matrixreader))  #dia chi tuong ung voi file dang xet
        
        an = next(matrixreader)
        temp_des = [np.float32(x) for x in an]

        for row in matrixreader:
            a = [np.float32(x) for x in row]
            temp_des= np.vstack((temp_des, a))

    # matches = bf.knnMatch(des,temp_des,k=2)
    matches = bf.match(des,temp_des)
    sum = 0
    for m in matches:
        sum = sum + m.distance 
    
    result.append((image_compare,sum))

def getKey(item):
    return item[1]


result = sorted(result, key=getKey)
for i, j in result:
    print(i)
    #finish = cv2.imread(i)
    #cv2.imshow("Result", finish) 
    #cv2.waitKey(0)













# Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.8*n.distance:
#         good.append([m])

# l2 = []
# with open('extract-feature/17.jpg', 'rb') as csvfile:
#     matrixreader = csv.reader(csvfile, delimiter=' ')
#     l2.append("".join(next(matrixreader)))
#     for row in matrixreader:
#         l2.append([float(x) for x in row])



# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# print len(matches)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.8*n.distance:
#         good.append([m])

# print len(good)
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

# plt.imshow(img3),plt.show()
# print len(des1[0])
# #print des1[0]
