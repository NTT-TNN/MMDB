import argparse as ap
import cv2
import os
import csv


train_path = "../data-train"

surf = cv2.xfeatures2d.SURF_create()

des_list = []
for img_path in os.listdir(train_path):
    img = cv2.imread(train_path + "/"+ img_path,0)
    print("load image: ", (train_path + "/"+ img_path))

    kp, des = surf.detectAndCompute(img,None)
    #des_list.append((img_path, des))

    with open('extract-feature/'+img_path, 'wb') as csvfile:
        matrixwriter = csv.writer(csvfile, delimiter=' ')
        a = str(train_path + "/"+ img_path)
        matrixwriter.writerow(a)
        for row in des:
            matrixwriter.writerow(row)





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
