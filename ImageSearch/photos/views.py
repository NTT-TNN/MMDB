from django.http import HttpResponse
from django.shortcuts import render
import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
import json
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views import View
from .forms import PhotoForm
from .models import Photo
import argparse as ap
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import csv

class Sorting(View):
    des2s=[]
    url_arr=[]
    train_path=""
    def get(self, request):
        self.train_path = "../ImageSearch/src/solution_2/"
        # training_names = os.listdir(train_path)
        # for training_name in training_names:
        #     self.des2s.append(np.load(train_path + training_name))
        #     self.url_arr.append(training_name.split(".")[0])
        photos_list = Photo.objects.all()
        print ("get upload ajax")
        return render(self.request, 'photos/basic_upload/sort_index.html', {'photos': photos_list})

    def post(self, request):
        print ("Finding image")
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            results=[]
            img_querry_path='../ImageSearch/media/'+photo.file.name
        #     img_search = cv2.imread(url_search,0)  # queryImage
        #     surf = cv2.xfeatures2d.SURF_create()
        #     kp1, des1 = surf.detectAndCompute(img_search, None)
        #
        #     cnt_arr = []
        #     # url_arr = []
        #     for index,des2 in enumerate(self.des2s):
        #         # print (des2)
        #         # des2 = np.load(train_path + training_name)
        #         FLANN_INDEX_KDTREE = 0
        #         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #         search_params = dict(checks=50)  # or pass empty dictionary
        #
        #         flann = cv2.FlannBasedMatcher(index_params, search_params)
        #
        #         matches = flann.knnMatch(des1, des2, k=2)
        #         print (index)
        #         print ("\n")
        #         matchesMask = [[0, 0] for i in range(len(matches))]
        #         cnt = 0
        #         for i, (m, n) in enumerate(matches):
        #             if m.distance < 0.7 * n.distance:
        #                 cnt = cnt + 1
        #                 matchesMask[i] = [1, 0]
        #         cnt_arr.append(cnt)
        #
        #         # print (training_name.split(".")[0])
        #     Z = [url for _, url in sorted(zip(cnt_arr, self.url_arr), reverse=True)]
        #     for url in Z[0:30]:
        #         result.append("image.orig/"+url+".jpg")
        #     context = {
        #         'result': result,
        #         'photo_file_url':photo.file.url
        #     }
        # else:
        #     context ={'is_valid': False}
        #     img_querry_path = "../data-test/0.jpg"
            img_vs_cluster = np.load("../ImageSearch/src/solution_2/img_vs_cluster.npy")
            voc = np.load("../ImageSearch/src/solution_2/voc.npy")
            img_paths = np.load("../ImageSearch/src/solution_2/img_paths.npy")

            print("Load xong du lieu tu file")

            surf = cv2.xfeatures2d.SURF_create()
            img_querry = cv2.imread(img_querry_path, 0)
            kp, des = surf.detectAndCompute(img_querry, None)

            words, dis_vs_clus = vq(des, voc)

            temp = np.zeros(len(voc))
            for w in words:
                temp[w] += 1

            result = []
            for i, image_vecto in enumerate(img_vs_cluster):
                dis = distance.euclidean(temp, image_vecto)
                result.append((img_paths[i], dis))
            def getKey(item):
                return item[1]

            result = sorted(result, key=getKey)
            for i, j in result:
                print (i[2:])
                results.append(i[2:])
            context = {
                'result': results,
                'photo_file_url': photo.file.url
            }
        return JsonResponse(context)

class BasicUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()
        print ("get upload ajax")
        return render(self.request, 'photos/basic_upload/index.html', {'photos': photos_list})

    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)

        if form.is_valid():
            photo = form.save()
            results = []
            print ("post ajax search")
        #     clf, classes_names, stdSlr, k, voc = joblib.load("../ImageSearch/src/train.txt")
        #     SIFT = cv2.xfeatures2d.SIFT_create()
        #     # des_ext = cv2.xfeatures2d.SURF_create()
            test_path = "../ImageSearch/media/"
        #
        #     des_list = []
        #     # print (photo.file.url)
            img_querry_path = os.path.join(test_path, photo.file.name)
        #     image_paths = [image_path]
        #     print (image_path)
        #     im = cv2.imread(image_path)
        #     kpts = SIFT.detect(im)
        #     kpts, des = SIFT.compute(im, kpts)
        #     des_list.append((image_path, des))
        #
        #     # Stack all the descriptors vertically in a numpy array
        #     descriptors = des_list[0][1]
        #     for image_path, descriptor in des_list[0:]:
        #         descriptors = np.vstack((descriptors, descriptor))
        #     test_features = np.zeros((len(image_paths), k), "float32")
        #     for i in range(len(image_paths)):
        #         words, distance = vq(des_list[i][1], voc)
        #         for w in words:
        #             test_features[i][w] += 1
        #
        #     # Perform Tf-Idf vectorization
        #     nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
        #     idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
        #
        #     # Scale the features
        #     test_features = stdSlr.transform(test_features)
        #
        #     # Perform the predictions
        #     predictions = [classes_names[i] for i in clf.predict(test_features)]
        #     # predictions = [key]
        #     # Visualize the results, if "visualize" flag set to true by the user
        #
        #     for image_path, prediction in zip(image_paths, predictions):
        #         # image = cv2.imread(image_path)
        #         print (prediction)
        #         results_path = "../ImageSearch/src/dataset/train/" + prediction
        #         results_name = os.listdir(results_path)
        #         for result_name in results_name:
        #             result_path = os.path.join(results_path, result_name)
        #             result_name = "/" + prediction + "/" + result_name;
        #             result.append(result_name)
        #     context = {
        #         'result': result,
        #         'photo_file_url':photo.file.url
        #     }
        # else:
        #     context ={'is_valid': False}
        feature_paths = "../ImageSearch/src/solution_1/extract-feature"
        # img_querry_path = "../data-test/0.jpg"

        surf = cv2.xfeatures2d.SURF_create()
        img_querry = cv2.imread(img_querry_path, 0)
        kp, des = surf.detectAndCompute(img_querry, None)
        bf = cv2.BFMatcher()

        result = []

        for feature_path in os.listdir(feature_paths):
            temp_des = []  # luu cac des duoc lay tu file
            print("Xu Ly Anh: ", feature_path)

            with open('../ImageSearch/src/solution_1/extract-feature/' + feature_path, 'rt', encoding="utf8") as csvfile:
                matrixreader = csv.reader(csvfile, delimiter=' ')
                image_compare = "".join(next(matrixreader))  # dia chi tuong ung voi file dang xet

                an = next(matrixreader)
                temp_des = [np.float32(x) for x in an]

                for row in matrixreader:
                    a = [np.float32(x) for x in row]
                    temp_des = np.vstack((temp_des, a))

            # matches = bf.knnMatch(des,temp_des,k=2)
            matches = bf.match(des, temp_des)
            sum = 0
            for m in matches:
                sum = sum + m.distance

            result.append((image_compare, sum))

        def getKey(item):
            return item[1]

        result = sorted(result, key=getKey)
        for i, j in result:
            results.append(i[2:])
        context = {
            'result': results,
            'photo_file_url': photo.file.url
        }
        return JsonResponse(context)

def clear_database(request):
    for photo in Photo.objects.all():
        photo.file.delete()
        photo.delete()
    return redirect(request.POST.get('next'))
