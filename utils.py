import os
from random import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

#get the path for the training data
data_dir = './data/train'

#define the different categories
categories = ['daisy', 'sunflower','rose', 'tulip','dandelion']

#create emty array for the data
data = []
def build_data():
    for category in categories:
        #connect paths for each category
        path = os.path.join(data_dir,category)
        #set the labes for each category (0-4)
        label = categories.index(category)

        for img_name in os.listdir(path):
            #get each individual image path
            image_path = os.path.join(path, img_name)
            #get the image of the specified path
            image = cv2.imread(image_path)

            try:
                #resize the images + Use RBG (if not negative colors)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                #append the transformed image to the dataset
                data.append([image, label])
                cv2.flip(image,0)
                data.append([image, label])
                cv2.flip(image,1)
                data.append([image, label])
                rotation(image,24)
                data.append([image, label])
            except Exception as e :
                     pass

        print("Number of " + str(category) +"s:" + str(len(data)))
        #store the data into a pickle file
        pik = open('data.pickle', 'wb')
        pickle.dump(data, pik)
        pik.close()

build_data()


def load_data():
    #open and load the picke file
    pick = open('data.pickle', 'rb')
    data = pickle.load(pick)
    pick.close()

    #shuffle the dataset
    np.random.shuffle(data)

    #create a feature and lables array
    feature = []
    labels = []

    #fill the arrays with respective data
    for img, label in data:
        feature.append(img)
        labels.append(label)

    #convert them to numpy arrays
    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)

    feature = feature/255.0

    #return an array with both newly created np arrays
    return[feature, labels]

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img