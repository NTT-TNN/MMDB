#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("train.txt")
# Create feature extraction and keypoint detector objects
fea_det = cv2.xfeatures2d.SURF_create()
des_ext = cv2.xfeatures2d.SURF_create()

test_path = "dataset/test/food/"
test_names = os.listdir(test_path)
image_paths=[]
for test_name in test_names:
    # List where all the descriptors are stored
    des_list = []
    image_path= os.path.join(test_path, test_name)
    image_paths=[image_path]
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))

        # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

        #
    test_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scale the features
    test_features = stdSlr.transform(test_features)

    # Perform the predictions
    predictions = [classes_names[i] for i in clf.predict(test_features)]

    # Visualize the results, if "visualize" flag set to true by the user

    for image_path, prediction in zip(image_paths, predictions):
        # image = cv2.imread(image_path)
        print (prediction)
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # pt = (0, 3 * image.shape[0] // 4)
        # cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        # cv2.imshow("Image", image)
        # cv2.waitKey(3000)
    



