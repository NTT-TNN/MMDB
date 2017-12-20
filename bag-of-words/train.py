#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Get the training classes names and store them in a list
train_path = "dataset/train/"
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Initiate SURF detector
SIFT = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = SIFT.detect(im)
    kpts, des = SIFT.compute(im, kpts)
    des_list.append((image_path, des))   
# print (des_list)
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) # Tách thành 100 cụm dựa trên phương sai

# np.savetxt('descriptors.txt', descriptors, delimiter=" ", fmt="%s")
# np.savetxt('voc.txt', voc, delimiter=" ", fmt="%s")
# print (np.max(descriptors))
# print (np.max(voc))

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32") # Tạo một array độ dài bằng độ dài image_paths và giá trị bằng 0
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    # print (words)
    for w in words:
        im_features[i][w] += 1
# print (im_features)

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Scaling the words
stdSlr = StandardScaler().fit(im_features)
# print (im_features)
# print ("\n")
im_features = stdSlr.transform(im_features)
# print (im_features)
# print ("\n")

# Train the Linear SVM
clf = LinearSVC()
print (im_features)
print (image_classes)
clf.fit(im_features, np.array(image_classes))
# Save to file
joblib.dump((clf, training_names, stdSlr, k, voc), "train.txt", compress=0)
    
