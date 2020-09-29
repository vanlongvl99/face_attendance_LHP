from django.shortcuts import render, redirect
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
# from . import dataset_fetch as df
# from . import cascade as casc
from PIL import Image
from numpy import expand_dims
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import Normalizer
from os import listdir
from tensorflow.keras.models import load_model
import datetime
import joblib
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
from mtcnn.mtcnn import MTCNN
import faiss
# from .settings import BASE_DIR
import datetime

import json
from sklearn.neighbors import KDTree

np.random.seed(100)


# set PATH
path_model_facenet =  '/home/vanlong/vanlong/cotai/facial_recog/ml/model_file/facenet_keras.h5'
path_weights_facenet =   '/home/vanlong/vanlong/cotai/facial_recog/ml/model_file/facenet_keras_weights.h5'

# image_train = 15
k = 3

detector = MTCNN()

# Load model facenet
model_facenet = load_model(path_model_facenet)
model_facenet.load_weights(path_weights_facenet )


path_train_data = "./dataset1/train_test_face/train10"
path_test_data = "./dataset1/train_test_face/test10"
path_face_image = "./dataset1/only_face"


# get the face embedding for one face
def transform(face_pixels):
    face_pixels = cv2.resize(face_pixels,(160,160))
    face_pixels = (face_pixels/255)
	# scale pixel values
    face = face_pixels.astype('float32')
    mean, std = face.mean(axis=1), face.std(axis=1)
    face = (face - mean) / std
    return face









def get_embedding_train_test(model_facenet, path_data):
    X_train = []
    y_label_train = []
    X_test = []
    y_label_test = []
    for index_name in listdir(path_data):
        index_arr = list(listdir(path_data + "/" + index_name))
        np.random.shuffle(index_arr)
        for i in range(len(index_arr)):
            # for image_train in range(5,60,3):
            if i < image_train:
                image = cv2.imread(path_face_image + "/" + index_name + "/" + index_arr[i])
                face = transform(image)
                X_train.append(face)
                y_label_train.append(int(index_name))
            else: 
                image = cv2.imread(path_face_image + "/" + index_name + "/" + index_arr[i]  )
                face = transform(image)
                X_test.append(face)
                y_label_test.append(int(index_name))
        print("done embedding:", index_name)
    X_train = model_facenet.predict(np.array(X_train))
    X_test = model_facenet.predict(np.array(X_test))
    return X_train, X_test, y_label_train, y_label_test


def get_embedding(model_facenet, path_data):
    X_train = []
    y_label = []
    for index_name in listdir(path_data):
        for file_name in listdir(path_data + "/" + index_name):
            image = cv2.imread(path_data + "/" + index_name + "/" + file_name)
            face = transform(image)
            X_train.append(face)
            y_label.append(int(index_name))
    X_train = model_facenet.predict(np.array(X_train))
    return X_train, y_label


# X_train, y_label = get_embedding(model_facenet, path_train_data)
# np.save("X_train_10.npy", X_train)
# np.save("y_label_10.npy", y_label)
# X_train = np.load("X_train_10.npy")
# y_label = np.load("y_label_10.npy")


# X_test, y_label_test = get_embedding(model_facenet, path_test_data) 
# np.save("X_test_10.npy", X_test)
# np.save("y_label_10_test.npy", y_label_test)
# X_test = np.load("X_test_10.npy")
# y_label_test = np.load("y_label_10_test.npy")
dict_check = {}
for image_train in range(38,60,3):
    X_train, X_test, y_label_train, y_label_test = get_embedding_train_test(model_facenet, path_face_image)
    tree = KDTree(X_train)
    print("loaded tree")
    distance, index = tree.query(X_test)
    print("index:",index.shape)
    wrong = 0
    total = 0
    for i in range(len(y_label_test)):
        # print(y_label_test[i],y_label[index[i,0]])
        if y_label_test[i] != y_label_train[index[i,0]]:
            wrong += 1
        total += 1
    print(1 - wrong/total)
    dict_check[str(image_train)] =  (1 - wrong/total)
    with open("image_train_" + str(image_train) + "_k_1.json", 'w') as json_file:
        json.dump(dict_check, json_file) 


def kdtree_query()

def detect():
    all_faces = []
    start_time = datetime.datetime.now()
    # DETECT FACE
    faces = detector.detect_faces(img)
    for person in faces:
        bounding_box = person["box"]
        im_crop = img[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
        if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
            # GET EMBEDDING FROM FACE
            im_crop = transform(im_crop)
            all_faces.append(im_crop)                
    all_faces = np.array(all_faces)
    print("shape:",all_faces.shape)
    if len(all_faces) > 0:
        im_embedding = model_facenet.predict(all_faces)
        distances, index = tree.query(im_embedding, k = 3)
        print(index)
