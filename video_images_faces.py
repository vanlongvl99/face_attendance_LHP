import tkinter as tk
from tkinter import Message, Text, messagebox
import cv2
import os
import time
import numpy as np
from sklearn.svm import SVC
# import joblib
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import datetime
from numpy import load
from os import listdir
import os.path
from datetime import datetime
import json
from mtcnn.mtcnn import MTCNN





path_raw_image = "./dataset1/raw"
path_face_image = "./dataset1/only_face"
path_file_json = 'index_to_name_test.json'

try:
    os.mkdir("dataset1")
except:
    pass

try:
    f = f = open(path_file_json,)
    index_to_name = json.load(f) 
except:
    index_to_name = {}
list_name = []
for index, name_and_number in index_to_name.items():
    list_name.append(name_and_number["name"])



#### Make GUI ####
window = tk.Tk()
window.title("Face_Recogniser")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Create dataset system",
    bg ="green", fg = "white", width = 50,
    height = 3, font = ('times', 30, 'bold'))
message.place(x = 200, y = 20)

lbl2 = tk.Label(window, text ="Name of new person",
width = 20, fg ="green", bg ="white",
height = 2, font =('times', 15, ' bold '))
lbl2.place(x = 400, y = 200)
txt2 = tk.Entry(window, width = 20,
bg ="white", fg ="green",
font = ('times', 15, ' bold '))
txt2.place(x = 700, y = 215)

lbl3 = tk.Label(window, text ="Number of new person",
width = 20, fg ="green", bg ="white",
height = 2, font =('times', 15, ' bold '))
lbl3.place(x = 400, y = 250)
txt3 = tk.Entry(window, width = 20,
bg ="white", fg ="green",
font = ('times', 15, ' bold '))
txt3.place(x = 700, y = 265)
############         #################







def TakeImages():

    name =(txt2.get())
    number_std = txt3.get()
    if name not in list_name:
        index_to_name[str(len(list_name))] = {"name":name,"number":number_std}
        list_name.append(name)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    index_name = len(list_name) - 1
    # print("start take image")
    try:
        os.mkdir(path_raw_image)
        print("make:", path_raw_image)
    except:
        pass
    try:
        os.mkdir(path_raw_image + "/" + str(index_name))
        os.mkdir(path_raw_image + "/" + str(index_name) + "/images")
        print("make:",path_raw_image + "/" + str(index_name) + "/images")
    except:
        pass
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter( path_raw_image+ "/" + str(index_name) + "/" + str(index_name) + "_video.avi", fourcc, 30.0, (640,480))
    
    # cam = cv2.VideoCapture("http://192.168.2.26:8080/video")
    # cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture(2)
    count = 0
    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame,(640,int(frame.shape[0]/frame.shape[1]*640)))
        cv2.imshow('frame_resize', frame)
        videoWriter.write(frame)
        key = cv2.waitKey(20)
        if count % 5 == 0 and count //5 < 100:
            cv2.imwrite(path_raw_image + "/" + str(index_name) + "/images/" + str(index_name) +"__" + str(count//5) + ".jpg", frame)
            print(path_raw_image + "/" + str(index_name) + "/images/" + str(index_name) +"__" + str(count//5) + ".jpg")
        count += 1
        if key  == ord('q'):
            print("Finished take",name)
            break
        # print(datetime.now())

    cam.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    with open(path_file_json, 'w') as json_file:
        json.dump(index_to_name, json_file)
        print(index_to_name)
        print("save_json_file")

def take_face_from_image():
    detector = MTCNN()
    try:
        os.mkdir(path_face_image)
    except:
        pass
# 
    for index_name in listdir(path_raw_image):
        try:
            os.mkdir(path_face_image + "/" + index_name)
            for file_name in listdir(path_raw_image + "/" + index_name + "/images"):
                start_time = datetime.now()
                image = cv2.imread(path_raw_image + "/" + index_name + "/images" + "/" + file_name)
                faces = detector.detect_faces(image)
                print("mtcnn:", datetime.now()- start_time)
                if len(faces) == 1:
                    for person in faces:
                        bounding_box = person['box']
                        im_crop = image[bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
                        if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                            cv2.imwrite(path_face_image + "/" + index_name + "/" + file_name, im_crop)
        except:
            pass
        
        
        print("finished:",index_name)
    print("finished")
    messagebox.showinfo(title="Thông báo", message="Hoàn tất take face!")





#Take image from camera
takeImg = tk.Button(window, text ="Take images",      
command = TakeImages, fg ="white", bg ="green",
# kich thuoc button  
width = 15, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Toa do button
takeImg.place(x = 150, y = 400) 

# # Take faces from images
takeImg = tk.Button(window, text ="Take face from image",      
command = take_face_from_image, fg ="white", bg ="green",
# kich thuoc button  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Toa do button
takeImg.place(x = 400, y = 400) 


# Quit GUI
quitWindow = tk.Button(window, text ="Quit",  
command = window.destroy, fg ="white", bg ="green",  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
quitWindow.place(x = 1050, y = 400) 
window.mainloop()