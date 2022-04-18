import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.datasets import mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import yaml
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import patches
from sklearn import preprocessing

import tensorflow as tf
import time
from sklearn.preprocessing import OneHotEncoder
from keras.layers.convolutional import Convolution2D
from sklearn import metrics

def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

data_path = 'C:/Users/grv/test/data'

with open(data_path + "train.yaml", "r") as test_file:
    train_yaml = yaml.safe_load(test_file)
with open(data_path + "test.yaml", "r") as test_file:
    test_yaml = yaml.safe_load(test_file)
print(len(train_yaml))
print(len(test_yaml))

train_df = pd.DataFrame(columns=('path', 'x_min', 'y_min', 'x_max', 'y_max', 'target'))

data_path = "C:/Users/grv/test/data"


print("data train")
x_train = []
counter = 0
for datapoint in train_yaml:
    counter += 1
    if counter % 1000 == 0:
        print("Processed {} of {}".format(counter, len(train_yaml)))
    pathname = datapoint["path"]
    image = cv2.imread(data_path + pathname)
    resized = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    if datapoint["boxes"]:
        for i in range(0, len(datapoint["boxes"])):
            xmin = datapoint["boxes"][i]['x_min']
            xmax = datapoint["boxes"][i]['x_max']
            ymin = datapoint["boxes"][i]['y_min']
            ymax = datapoint["boxes"][i]['y_max']

            if datapoint["boxes"][i]:
                if datapoint["boxes"][i]['label'].startswith("Green"):
                    classname = "Green"
                elif datapoint["boxes"][i]['label'].startswith("Red"):
                    classname = "Red"
                elif datapoint["boxes"][i]['label'].startswith("Yellow"):
                    classname = "Yellow"
                elif datapoint["boxes"][i]['label'].startswith("off"):
                    classname = "off"
                else:
                    print(datapoint["boxes"][i]['label'])
                    print("something wrong")
                dict1 = {'path': pathname, 'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax,
                         'target': classname}
                train_df = train_df.append(dict1, ignore_index=True)
                x_train.append(resized)
print("Processed {} of {}".format(counter, len(train_yaml)))

print("Test data")
test_df = pd.DataFrame(columns=('path', 'x_min', 'y_min', 'x_max', 'y_max', 'target'))
x_test = []
counter = 0
for datapoint in test_yaml:
    counter += 1
    if counter % 1000 == 0:
        print("Processed {} of {}".format(counter, len(test_yaml)))
    if datapoint["boxes"]:
        pathname = datapoint["path"]
        tmp = os.path.basename(pathname)
        pathname = data_path + 'rgb/test/' + tmp
        image = cv2.imread(pathname)
        resized = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        for i in range(0, len(datapoint["boxes"])):
            xmin = datapoint["boxes"][i]['x_min']
            xmax = datapoint["boxes"][i]['x_max']
            ymin = datapoint["boxes"][i]['y_min']
            ymax = datapoint["boxes"][i]['y_max']

            if datapoint["boxes"][i]:
                if datapoint["boxes"][i]['label'].startswith("Green"):
                    classname = "Green"
                elif datapoint["boxes"][i]['label'].startswith("Red"):
                    classname = "Red"
                elif datapoint["boxes"][i]['label'].startswith("Yellow"):
                    classname = "Yellow"
                elif datapoint["boxes"][i]['label'].startswith("off"):
                    classname = "off"
                else:
                    print("something wrong")

                dict1 = {'path': pathname, 'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax,
                         'target': classname}
                test_df = test_df.append(dict1, ignore_index=True)
                x_test.append(resized)
print("Processed {} of {}".format(counter, len(test_yaml)))


x_train_ = np.array(x_train)
X_test_ = np.array(x_test)
x_train_ = x_train_.astype('float32')
X_test_ = X_test_.astype('float32')
x_train_ /= 255
X_test_ /= 255
print(x_train_.shape)
print(X_test_.shape)

y_trian_ = OneHotEncoder().fit_transform(train_df[['target']]).toarray()
y_test_ = OneHotEncoder().fit_transform(test_df[['target']]).toarray()

print(y_trian_.shape)
print(y_test_.shape)

NUM_CHANNELS = 3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_CLASSES = 4

model = Sequential()
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)))
model.add((Conv2D(32,activation='relu',kernel_size=(3,3))))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= 'adam',
              metrics=['accuracy'])
model.summary()

batch_size = 128
epochs = 10

model.fit(x_train_, y_trian_,
          batch_size=batch_size,
          epochs=epochs)
score = model.evaluate(X_test_, y_test_, verbose=0)

print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))

y_test = np.argmax(y_test_,axis=1)
model_pred = model.predict(X_test_)
pred = np.argmax(model_pred,axis=1)


score = metrics.accuracy_score(y_test, pred)
print('Accuracy: {}'.format(score))

r = pd.DataFrame( { 'y': y_test, "pred": pred})
r.to_csv("output.csv")