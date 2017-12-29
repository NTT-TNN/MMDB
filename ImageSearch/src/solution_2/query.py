
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

img_querry_path = "../data-test/0.jpg"
img_vs_cluster = np.load("img_vs_cluster.npy")
voc = np.load("voc.npy")
img_paths = np.load("img_paths.npy")

print("Load xong du lieu tu file")

surf = cv2.xfeatures2d.SURF_create()
img_querry = cv2.imread(img_querry_path,0) 
kp, des = surf.detectAndCompute(img_querry,None)

words, dis_vs_clus = vq(des, voc)

temp = np.zeros(len(voc))
for w in words:
    temp[w] += 1


result = []
for i, image_vecto in enumerate(img_vs_cluster):
    dis = distance.euclidean(temp,image_vecto)
    result.append((img_paths[i], dis))

#print result


def getKey(item):
    return item[1]

result = sorted(result, key=getKey)
for i, j in result:
    print(i)
    # finish = cv2.imread(i)
    # cv2.imshow("Result", finish) 
    # cv2.waitKey(0)
