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
                print ((i / len(img_vs_cluster)), "%")
                dis = distance.euclidean(temp, image_vecto)
                result.append((img_paths[i], dis))
            def getKey(item):
                return item[1]

            result = sorted(result, key=getKey)
            count=0;
            for i, j in result:
                count=count+1
                results.append(i[2:])
                if count==100:
                    break
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

    def post(self, request):            # matches = bf.knnMatch(des,temp_des,k=2)

        form = PhotoForm(self.request.POST, self.request.FILES)

        if form.is_valid():
            photo = form.save()
            results = []
            print ("post ajax search")
            test_path = "../ImageSearch/media/"
            img_querry_path = os.path.join(test_path, photo.file.name)
        feature_paths = "../ImageSearch/src/solution_1/extract-feature"
        surf = cv2.xfeatures2d.SURF_create()
        img_querry = cv2.imread(img_querry_path, 0)
        kp, des = surf.detectAndCompute(img_querry, None)
        bf = cv2.BFMatcher()

        result = []
        p=0;
        for feature_path in os.listdir(feature_paths):
            p=p+1
            print ((p/len(os.listdir(feature_paths))),"%")
            # print ("%")
            # print("Xu Ly Anh: ", feature_path)
            with open('../ImageSearch/src/solution_1/extract-feature/' + feature_path, 'rt', encoding="utf8") as csvfile:
                matrixreader = csv.reader(csvfile, delimiter=' ')
                image_compare = "".join(next(matrixreader))  # dia chi tuong ung voi file dang xet

                an = next(matrixreader)
                temp_des = [np.float32(x) for x in an]

                for row in matrixreader:
                    a = [np.float32(x) for x in row]
                    temp_des = np.vstack((temp_des, a))
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
