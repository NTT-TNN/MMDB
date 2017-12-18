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


class BasicUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()
        print ("get upload ajax")
        return render(self.request, 'photos/basic_upload/index.html', {'photos': photos_list})

    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)

        if form.is_valid():
            photo = form.save()
            result = []
            print ("post ajax search")
            clf, classes_names, stdSlr, k, voc = joblib.load("/home/thao-nt/Desktop/MMDB/MMDB/ImageSearch/searchApp/train.txt")
            fea_det = cv2.xfeatures2d.SURF_create()
            des_ext = cv2.xfeatures2d.SURF_create()
            print (k)
            test_path = "/home/thao-nt/Desktop/MMDB/MMDB/ImageSearch/media/"

            des_list = []
            print (photo.file.url)
            image_path = os.path.join(test_path, photo.file.name)
            image_paths = [image_path]
            print (image_path)
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
            # predictions = [key]
            # Visualize the results, if "visualize" flag set to true by the user

            for image_path, prediction in zip(image_paths, predictions):
                # image = cv2.imread(image_path)
                print (prediction)
                results_path = "/home/thao-nt/Desktop/MMDB/MMDB/ImageSearch/searchApp/dataset/train/" + prediction
                results_name = os.listdir(results_path)
                for result_name in results_name:
                    result_path = os.path.join(results_path, result_name)
                    result_name = "/" + prediction + "/" + result_name;
                    result.append(result_name)
            context = {
                'result': result,
                'photo_file_url':photo.file.url
            }
        else:
            context ={'is_valid': False}

        return JsonResponse(context)

def clear_database(request):
    for photo in Photo.objects.all():
        photo.file.delete()
        photo.delete()
    return redirect(request.POST.get('next'))
