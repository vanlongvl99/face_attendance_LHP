from django.shortcuts import render, redirect
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc
from PIL import Image
from numpy import expand_dims
from time import time
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
from .settings import BASE_DIR
import datetime
import json


detector = MTCNN()


# set PATH
path_model_facenet = BASE_DIR +  '/ml/model_file/facenet_keras.h5'
path_weights_facenet = BASE_DIR +  '/ml/model_file/facenet_keras_weights.h5'
path_of_data = BASE_DIR+'/dataset1/only_face/'
path_file_json = BASE_DIR + '/index_to_name_test.json'
f = open(path_file_json,)
index_to_label = json.load(f) 



# Load model facenet
mode_facenet = ""
model_facenet = load_model(path_model_facenet)
model_facenet.load_weights(path_weights_facenet )

# Create your views here.
def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')


def create_dataset(request):
    userId = request.POST['userId']
    # Detect face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)
    user_name = userId
    # Our dataset naming counter
    sampleNum = 0
    try:
        os.mkdir(path_of_data + user_name)
    except:
        pass
    # Capturing the faces one by one and detect the faces and showing it on the window
    count = 0
    while(True):
        # Capturing the image
        #cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        faces = detector.detect_faces(img)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        print(len(faces))
        for person in faces:
            bounding_box = person['box']
            im_crop = img[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                cv2.rectangle(img,(bounding_box[0],bounding_box[1]),(bounding_box[0] + bounding_box[2],bounding_box[1] + bounding_box[3]),(0,155,255),3)
                count += 1
                cv2.imwrite(path_of_data + user_name + "/" + user_name + str(count) + ".jpg", im_crop)
            
        cv2.imshow('frame', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
           break
            # break if the sample number is more than 300
        elif count == 30:
            break
    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')

# get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = cv2.resize(face_pixels,(160,160))
    face_pixels = (face_pixels/255)
	# scale pixel values
    face = face_pixels.astype('float32')
    mean, std = face.mean(axis=1), face.std(axis=1)
    face = (face - mean) / std
    samples = expand_dims(face, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def load_data(path):
    index_to_label = {}
    cnt = 0
    X_train = []
    y_labels_train = []
    for forder_name in listdir(path):
        index_to_label[cnt] = forder_name
        count = 0
        for file_name in listdir(path + '/' + forder_name):
            path_of_image = path + '/' + forder_name + '/' + file_name
            image = cv2.imread(path_of_image)
            count += 1
            if count == 100:
                break
            image = cv2.resize(image,(160,160))
            X_train.append(image)
            y_labels_train.append(cnt)
        cnt += 1
        print( forder_name,count)
    print(index_to_label)
    return np.array(X_train), np.array(y_labels_train), index_to_label

def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    from PIL import Image
    from sklearn.svm import SVC
    #Path of the samples
    X_train, y_labels_train, index_to_label = load_data(path_of_data)
    # convert each face in the train set to an embedding
    newX_train = list()
    for face_pixels in X_train:
    	embedding = get_embedding(model_facenet, face_pixels)
    	newX_train.append(embedding)
    newX_train = np.array(newX_train)
    print(newX_train.shape)
    # normalize input vectors
    # fit model
    model_svm = SVC(kernel='linear', probability=True)
    print("shape incoder:",newX_train.shape)
    model_svm.fit(newX_train, y_labels_train)
    # predict
    yhat_train = model_svm.predict(newX_train)
    print(datetime.datetime.now(), "finished")
    #save model svm
    filename_svm_model = BASE_DIR + "/ml/model_file/svm_model.sav"
    joblib.dump(model_svm, filename_svm_model)

    return redirect('/')




def detect(request):
    #connect with google sheet
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("attendancelhp.json", scope)
    client = gspread.authorize(creds)
    # set position of this week
    # index_row = 7
    today = str(datetime.date.today())
    sheet = client.open("test_attendance").sheet1  # Open the spreadhseet
    col2_values = sheet.col_values(2)  # Get a specific column
    row1_values = sheet.row_values(1)
    # print("col2_values", col2_values)
    # print("row1_values", row1_values)
    index_row = len(row1_values)
    if row1_values[len(row1_values)-1] != today:
        sheet.update_cell(1, len(row1_values) + 1,today)
        index_row += 1
    filename_svm_model = BASE_DIR + "/ml/model_file/svm_model.sav"
    loaded_model_SVM = joblib.load(filename_svm_model)
    cam = cv2.VideoCapture(0)
    print(index_to_label)
    print("detect")
    while(True):
        ret, img = cam.read()
        faces = detector.detect_faces(img)
        for person in faces:
            bounding_box = person["box"]
            im_crop = img[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                im_embedding = get_embedding(model_facenet, im_crop)
                im_embedding = expand_dims(im_embedding,axis = 0)
                pre_face = loaded_model_SVM.predict_proba(im_embedding)[0]
                print(datetime.datetime.now(), "end")
                for i in range(len(index_to_label)):
                    name_and_class = index_to_label[str(i)]
                    print(name_and_class["name"] + ':','{0:0.2f}'.format(pre_face[i]))
                max_index = np.argmax(pre_face)

                # print(index_to_label[max_index],pre_face[max_index], "end\n")
                cv2.rectangle(img,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                if pre_face[max_index] > 0.75:
                    print("max index:",max_index)
                    name_and_class = index_to_label[str(max_index)]
                    cv2.putText(img, name_and_class["name"] + ': ' + str('{0:0.2f}'.format(pre_face[max_index])), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)
                    if name_and_class["name"] not in col2_values:
                        sheet.update_cell(len(col2_values) + 1, 2, name_and_class["name"])
                        sheet.update_cell(len(col2_values) + 1, 3, name_and_class["class"])
                        col2_values.append(name_and_class["name"])
                    index_col = col2_values.index(name_and_class["name"]) + 1
                    # row_check_values = sheet.row_values(index_col)
                    if len(sheet.row_values(index_col)) < len(sheet.row_values(1)):
                        now = datetime.datetime.now()
                        time_true = datetime.datetime(now.year,now.month,now.day,now.hour + 7, now.minute,now.second)
                        sheet.update_cell(index_col, index_row, str(time_true))

        cv2.imshow('image',img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')

def eigenTrain(request):
    path = BASE_DIR+'/ml/dataset'

    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= df.getImagesWithID(path)
    # print 'features'+str(faces.shape[1])
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    #print ">>>>>>>>>>>>>>> "+str(y_test.size)
    n_classes = y_test.size
    target_names = ['Manjil Tamang', 'Marina Tamang','Anmol Chalise']
    n_components = 15
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()

    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    svm_pkl_filename = BASE_DIR+'/ml/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()



    pca_pkl_filename = BASE_DIR+'/ml/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

    return redirect('/')


def detectImage(request):
    # index_to_label = {0: '419', 1: '836', 2: '556', 3: '897', 4: '593', 5: '61', 6: '97', 7: '513', 8: '175', 9: '855', 10: '992', 11: '828', 12: '977', 13: '680', 14: '65', 15: '497', 16: '569', 17: '877', 18: '675', 19: '616', 20: '910', 21: '774', 22: '399', 23: '789', 24: '179', 25: '510', 26: '148', 27: '514', 28: '155', 29: '770', 30: '940', 31: '826', 32: '337', 33: '908', 34: '465', 35: '578', 36: '165', 37: '272', 38: '713', 39: '550', 40: '963', 41: '753', 42: '236', 43: '692', 44: '780', 45: '568', 46: '523', 47: 'van_long', 48: '816', 49: '29', 50: '12', 51: '32', 52: '949', 53: '66', 54: '660', 55: '92', 56: '932', 57: '53', 58: '702', 59: '923', 60: '668', 61: '176', 62: '943', 63: '276', 64: '739', 65: '817', 66: '841', 67: '512', 68: '645', 69: '449', 70: '547', 71: '260', 72: '471', 73: '538', 74: '402', 75: '225', 76: '282', 77: '827', 78: '169', 79: '980', 80: '361', 81: '829', 82: '938', 83: '252', 84: '316', 85: '131', 86: '804', 87: '70', 88: '195', 89: '577', 90: '620', 91: '727', 92: '858', 93: '565', 94: '119', 95: '703', 96: '339', 97: '630', 98: '989', 99: '531', 100: '706', 101: '68', 102: '78', 103: '953', 104: '610', 105: '736', 106: '25', 107: '705', 108: '677', 109: '990', 110: '697', 111: '127', 112: '416', 113: '206', 114: '4', 115: '969', 116: '485', 117: '935', 118: '108', 119: '117', 120: '1004', 121: '100', 122: '985', 123: '488', 124: '221', 125: '79', 126: '218', 127: '5', 128: '931', 129: '85', 130: '914', 131: '243', 132: '101', 133: '790', 134: '20', 135: '483', 136: '891', 137: '475', 138: '458', 139: '426', 140: '862', 141: '482', 142: '936', 143: '45', 144: '50', 145: '147', 146: '617', 147: '954', 148: '59', 149: '920', 150: '111', 151: 'trong_nghia', 152: '194', 153: '748', 154: '251', 155: '450', 156: '191', 157: '498', 158: '993', 159: '262', 160: '217', 161: '456', 162: '383', 163: '915', 164: '105', 165: '331', 166: '870', 167: '1017', 168: '479', 169: '226', 170: '499', 171: '595', 172: '762', 173: '983', 174: '852', 175: '409', 176: '785', 177: '247', 178: '63', 179: '533', 180: '928', 181: '384', 182: '292', 183: '22', 184: '758', 185: '138', 186: '582', 187: '269', 188: '132', 189: '746', 190: '491', 191: '89', 192: '278', 193: '751', 194: '401', 195: '325', 196: '678', 197: '28', 198: '313', 199: '397', 200: '833', 201: '641', 202: '580', 203: '44', 204: '781', 205: '279', 206: '1002', 207: '239', 208: '728', 209: '104', 210: '484', 211: '109', 212: '952', 213: '227', 214: '655', 215: '432', 216: '966', 217: '163', 218: '357', 219: '941', 220: '594', 221: '303', 222: '711', 223: '438', 224: '554', 225: '354', 226: '856', 227: '433', 228: '290', 229: '873', 230: '177', 231: '687', 232: 'bich_lan', 233: '224', 234: '21', 235: '372', 236: '965', 237: '328', 238: '378', 239: '626', 240: '180', 241: '144', 242: '731', 243: '398', 244: '545', 245: '912', 246: '123', 247: '786', 248: '114', 249: '506', 250: '893', 251: '302', 252: '540', 253: '1019', 254: '94', 255: '43', 256: '496', 257: '882', 258: '890', 259: '418', 260: '849', 261: '544', 262: '874', 263: '107', 264: '369', 265: '546', 266: '270', 267: '454', 268: '552', 269: '128', 270: '334', 271: '82', 272: '145', 273: '603', 274: '636', 275: '129', 276: '517', 277: '346', 278: '214', 279: '729', 280: '268', 281: '682', 282: '404', 283: '810', 284: '327', 285: '951', 286: '642', 287: '534', 288: '881', 289: '960', 290: '41', 291: '310', 292: '725', 293: '666', 294: '40', 295: '974', 296: '650', 297: '495', 298: '651', 299: '363', 300: '36', 301: '698', 302: '403', 303: '737', 304: '749', 305: '293', 306: '375', 307: '775', 308: '535', 309: '883', 310: '755', 311: '428', 312: '130', 313: '296', 314: '995', 315: '756', 316: '709', 317: '500', 318: '900', 319: '396', 320: '543', 321: '892', 322: '627', 323: '216', 324: '436', 325: '442', 326: '644', 327: '612', 328: '994', 329: '187', 330: '185', 331: '249', 332: '400', 333: '851', 334: '791', 335: '142', 336: '77', 337: '661', 338: '679', 339: '26', 340: '17', 341: '913', 342: '991', 343: '86', 344: '371', 345: '30', 346: '773', 347: '381', 348: '141', 349: '700', 350: '324', 351: '838', 352: '273', 353: '888', 354: '477', 355: '48', 356: '997', 357: '978', 358: '1014', 359: '242', 360: '220', 361: '184', 362: '283', 363: '934', 364: '875', 365: '884', 366: '898', 367: '509', 368: '503', 369: '16', 370: '166', 371: '605', 372: '905', 373: '979', 374: '417', 375: '518', 376: '99', 377: '341', 378: '56', 379: '689', 380: '10', 381: '151', 382: '629', 383: '899', 384: '295', 385: '1', 386: '662', 387: '455', 388: '298', 389: '9', 390: '672', 391: '248', 392: '696', 393: '168', 394: '412', 395: '481', 396: '443', 397: '286', 398: '654', 399: '656', 400: '750', 401: '982', 402: '579', 403: '219', 404: '571', 405: '171', 406: '735', 407: '784', 408: '757', 409: '765', 410: '368', 411: '541', 412: '776', 413: '81', 414: '115', 415: '929', 416: '84', 417: '760', 418: '701', 419: '11', 420: '199', 421: '440', 422: '767', 423: '437', 424: '207', 425: '721', 426: '37', 427: '766', 428: '551', 429: '526', 430: '814', 431: '606', 432: '895', 433: '558', 434: '805', 435: '670', 436: '338', 437: '193', 438: '581', 439: '624', 440: '906', 441: '933', 442: '23', 443: '390', 444: '90', 445: '613', 446: '253', 447: '597', 448: '47', 449: '609', 450: '532', 451: '637', 452: '18', 453: '106', 454: '924', 455: '467', 456: '894', 457: '961', 458: '152', 459: '658', 460: '1018', 461: '413', 462: '439', 463: '486', 464: '1006', 465: '792', 466: '232', 467: '740', 468: '380', 469: '263', 470: '228', 471: '393', 472: '31', 473: '694', 474: '937', 475: '840', 476: '999', 477: '389', 478: '186', 479: '422', 480: '887', 481: '330', 482: '309', 483: '685', 484: '600', 485: '3', 486: '519', 487: '527', 488: '351', 489: '183', 490: '808', 491: '246', 492: '304', 493: '353', 494: '13', 495: '259', 496: '822', 497: '699', 498: '8', 499: '590', 500: '859', 501: '492', 502: '821', 503: '649', 504: '319', 505: '349', 506: '919', 507: '602', 508: '254', 509: '942', 510: '1000', 511: '971', 512: '289', 513: '863', 514: '406', 515: '134', 516: '643', 517: '466', 518: '823', 519: '948', 520: '521', 521: '669', 522: '975', 523: '370', 524: '229', 525: '233', 526: '693', 527: '688', 528: '998', 529: '926', 530: '939', 531: '799', 532: '60', 533: '118', 534: '345', 535: '570', 536: '463', 537: '350', 538: '170', 539: '844', 540: '1016', 541: '1001', 542: '200', 543: '2', 544: '352', 545: '457', 546: '149', 547: '839', 548: '230', 549: '564', 550: '385', 551: '584', 552: '769', 553: '981', 554: '7', 555: '622', 556: '308', 557: '405', 558: '516', 559: '505', 560: '909', 561: '854', 562: '140', 563: '885', 564: '306', 565: '652', 566: '135', 567: '798', 568: '853', 569: '377', 570: '34', 571: '825', 572: '407', 573: '264', 574: '761', 575: '15', 576: '764', 577: '608', 578: '868', 579: '946', 580: '818', 581: '388', 582: '812', 583: '667', 584: '238', 585: '619', 586: '87', 587: '274', 588: '358', 589: '561', 590: '511', 591: '446', 592: '312', 593: '589', 594: '723', 595: '611', 596: '136', 597: '153', 598: '904', 599: '566', 600: '14', 601: '528', 602: '621', 603: '69', 604: '414', 605: '64', 606: '429', 607: '245', 608: '460', 609: '362', 610: '676', 611: '420', 612: '75', 613: '843', 614: '441', 615: '277', 616: '208', 617: '848', 618: '553', 619: '502', 620: '732', 621: '196', 622: '837', 623: '51', 624: '596', 625: '663', 626: '447', 627: '880', 628: '801', 629: '539', 630: '323', 631: '156', 632: '730', 633: '726', 634: '867', 635: '188', 636: '359', 637: '742', 638: '103', 639: '137', 640: '772', 641: '583', 642: '301', 643: '373', 644: '724', 645: '464', 646: '489', 647: '720', 648: '344', 649: '857', 650: '88', 651: '573', 652: '865', 653: '811', 654: '98', 655: '871', 656: '549', 657: '24', 658: '972', 659: '38', 660: '95', 661: '586', 662: '212', 663: '62', 664: '235', 665: '623', 666: '275', 667: '256', 668: '950', 669: '717', 670: '231', 671: '879', 672: '715', 673: '39', 674: '525', 675: '300', 676: '1013', 677: '379', 678: '240', 679: '198', 680: '562', 681: '1012', 682: '927', 683: '162', 684: '530', 685: '411', 686: '307', 687: '474', 688: '976', 689: '646', 690: '638', 691: '958', 692: '121', 693: '1005', 694: '508', 695: '889', 696: '201', 697: '294', 698: '435', 699: '159', 700: '181', 701: '197', 702: '367', 703: '842', 704: '712', 705: '864', 706: '598', 707: '472', 708: '964', 709: '832', 710: '835', 711: '444', 712: '19', 713: '986', 714: '752', 715: '143', 716: '461', 717: '281', 718: '55', 719: '601', 720: '695', 721: '778', 722: '861', 723: '683', 724: '265', 725: '945', 726: '234', 727: '876', 728: '957', 729: '1003', 730: '634', 731: '710', 732: '473', 733: '618', 734: '633', 735: '317', 736: '779', 737: '271', 738: '410', 739: '639', 740: '91', 741: '591', 742: '267', 743: '559', 744: '395', 745: '329', 746: '299', 747: '322', 748: '493', 749: '896', 750: '747', 751: '819', 752: '122', 753: 'manh_hung', 754: '305', 755: '567', 756: '968', 757: '1011', 758: '120', 759: '524', 760: '408', 761: '763', 762: '374', 763: '794', 764: '382', 765: '154', 766: '332', 767: '1015', 768: '973', 769: '759', 770: '922', 771: '356', 772: '850', 773: '190', 774: '860', 775: '916', 776: '690', 777: '647', 778: '360', 779: '557', 780: '846', 781: '261', 782: '803', 783: '469', 784: '215', 785: '592', 786: '318', 787: '58', 788: '189', 789: '869', 790: '213', 791: '797', 792: '800', 793: '116', 794: '480', 795: '815', 796: '124', 797: '415', 798: '1009', 799: '507', 800: '902', 801: '536', 802: '738', 803: '297', 804: '674', 805: '427', 806: '722', 807: '178', 808: '771', 809: '244', 810: '376', 811: '160', 812: '719', 813: '102', 814: '768', 815: '287', 816: '944', 817: '167', 818: '575', 819: '684', 820: '340', 821: '632', 822: '172', 823: '6', 824: '453', 825: '33', 826: '872', 827: '315', 828: '959', 829: '490', 830: '67', 831: '451', 832: '280', 833: '192', 834: '421', 835: '205', 836: '347', 837: '487', 838: '314', 839: '257', 840: '54', 841: '266', 842: '587', 843: '202', 844: '476', 845: '211', 846: '599', 847: '628', 848: '434', 849: '478', 850: '504', 851: '321', 852: '71', 853: '520', 854: '96', 855: '607', 856: '793', 857: '49', 858: '343', 859: '635', 860: '631', 861: '704', 862: '74', 863: '113', 864: '681', 865: '424', 866: '659', 867: '585', 868: 'duc_anh', 869: '907', 870: '326', 871: '576', 872: '448', 873: '604', 874: '847', 875: '164', 876: '423', 877: '157', 878: '42', 879: '809', 880: '76', 881: '515', 882: '430', 883: '366', 884: '501', 885: '962', 886: '222', 887: '653', 888: '125', 889: '241', 890: '671', 891: '718', 892: '258', 893: '806', 894: '209', 895: '996', 896: '640', 897: '788', 898: '73', 899: '173', 900: '664', 901: '27', 902: '336', 903: '802', 904: '342', 905: '947', 906: '716', 907: '445', 908: '255', 909: '462', 910: '917', 911: '133', 912: '560', 913: '459', 914: '921', 915: '824', 916: '210', 917: '83', 918: '831', 919: '394', 920: '93', 921: '537', 922: '743', 923: '150', 924: '139', 925: '673', 926: '126', 927: '46', 928: '431', 929: '615', 930: '563', 931: '1008', 932: '572', 933: '250', 934: '795', 935: '708', 936: '365', 937: '813', 938: '284', 939: '686', 940: '529', 941: '204', 942: '930', 943: '355', 944: '903', 945: '714', 946: '72', 947: '387', 948: '548', 949: '796', 950: '425', 951: '392', 952: '35', 953: '783', 954: '158', 955: '734', 956: '285', 957: '733', 958: '614', 959: '648', 960: '494', 961: '112', 962: '820', 963: '348', 964: '386', 965: '657', 966: '203', 967: '987', 968: '782', 969: '845', 970: '452', 971: '830', 972: '52', 973: '988', 974: '522', 975: '1010', 976: '542', 977: '754', 978: '333', 979: '744', 980: '237', 981: '311', 982: '174', 983: '741', 984: '335', 985: '574', 986: '468', 987: '886', 988: '161', 989: '291', 990: '967', 991: '691', 992: '625', 993: '777', 994: '918', 995: '146', 996: '588', 997: '57', 998: '901', 999: '866', 1000: '555', 1001: '745', 1002: '707', 1003: '911', 1004: '834', 1005: '878', 1006: '80', 1007: '470', 1008: '364', 1009: '807', 1010: '391', 1011: '110', 1012: '665', 1013: '288', 1014: '320', 1015: '223', 1016: '925', 1017: '182', 1018: '787'}

    k = 5
    loaded_model_SVM = joblib.load(svm_file)
    detector = MTCNN()
    # X_data, y_labels_train = get_data_knn("./dataset/face_data") 
    X_data = np.load(BASE_DIR + "/X_train_embedding.npy")
    y_labels_train = np.load(BASE_DIR + "/y_train.npy")
    ### faiss
    d = 128
    search_model = faiss.IndexFlatL2(d)
    search_model.add(X_data)
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture('http://192.168.1.16:8080/video')
    while True:
        ret, image = cam.read()
        # mtcnn detection
        # start_time = time.time()
        image = cv2.flip(image, 1)
        image = cv2.resize(image,(640,int(image.shape[0]/image.shape[1]*640)))
        faces = detector.detect_faces(image)
        print("\nmtcnn:",time.time()-start_time)
        for person in faces:
            bounding_box = person['box']
            # Specifying the coordinates of the image as well
            im_crop = image[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                # facemet embedding
                im_embedding = get_embedding(model_facenet, im_crop)
                # print("facenet:",time.time()-start_time)
                print(X_data.shape)
                print(im_embedding.shape)
                count_k_min_labels = np.zeros(len(index_to_label))

                ###### faiss #################
                D, I = search_model.search(np.array([im_embedding]), k)
                print("D:",D)
                print("I:",I)
                predictions = []
                for i in range(len(I[0])):
                    la = int(y_labels_train[I[0][i]])
                    dis = D[0][i]
                    predictions.append(la)
                print("\nresult:",predictions)
                for x in predictions:
                    count_k_min_labels[x] += 1
                print(datetime.datetime.now(), "end")
                # show result
                cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                cv2.putText(image, index_to_label[np.argmax(count_k_min_labels)] , (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)
        cv2.putText(image,"fps: " + str(round(1/(time.time()-start_time),2)) , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)
        cv2.imshow('image',image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')




def detectImage1(request):
    userImage = request.FILES['userImage']

    svm_pkl_filename = BASE_DIR+'/ml/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    #print "Loaded SVM model :: ", svm_model

    pca_pkl_filename =  BASE_DIR+'/ml/serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    #print pca

    '''
    First Save image as cv2.imread only accepts path
    '''
    im = Image.open(userImage)
    #im.show()
    imgPath = BASE_DIR+'/ml/uploadedImages/'+str(userImage)
    im.save(imgPath, 'JPEG')

    '''
    Input Image
    '''
    try:
        inputImg = casc.facecrop(imgPath)
        inputImg.show()
    except :
        print("No face detected, or image not recognized")
        return redirect('/error_image')

    imgNp = np.array(inputImg, 'uint8')
    #Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    #print imgFlatten
    #print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    #print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    # print pred[0]

    return redirect('/records/details/'+str(pred[0]))
