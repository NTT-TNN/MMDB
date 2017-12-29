

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

train_path = "../data-train"

surf = cv2.xfeatures2d.SURF_create()
sum_image = 0
des_list = []
img_paths = []

for path in os.listdir(train_path):
    image_path = train_path + "/"+ path
    img = cv2.imread(image_path,0)
    img_paths.append(image_path)
    print ("load image: ", image_path)

    kp, des = surf.detectAndCompute(img,None)

    des_list.append((image_path,des))
    sum_image += 1

des_train = des_list[0][1]
for path, temp_des in des_list[1:]:
    des_train = np.vstack((des_train, temp_des))

print ("Dang phan cum tat ca cac features.......")
sum_cluster = 1000
voc, distortion = kmeans(des_train, sum_cluster, 1)

print ("Tao xong Voc")
img_vs_cluster = np.zeros((sum_image, sum_cluster))

for i in range(sum_image):
    words, dis_vs_clus = vq(des_list[i][1], voc)
    print ("do dai words: ", len(words))
    print ("Mang phan cum cho cac feature cua image ", des_list[i][0] ," la: ", words)
    for w in words:
        img_vs_cluster[i][w] += 1

print ("\nMang tan suat cua image va cluster  ", img_vs_cluster)

# so image co chua cluster
df = np.sum( (img_vs_cluster > 0) * 1, axis = 0) 
print ("\ndf: ", df)

idf = np.array(np.log(sum_image / df), 'float32')
print ("\nidf: ", idf)


img_vs_cluster = img_vs_cluster*idf

np.save("img_vs_cluster.npy", img_vs_cluster)
np.save("voc.npy", voc)
np.save("img_paths.npy", img_paths)

print ("Da luu ra file!!")
