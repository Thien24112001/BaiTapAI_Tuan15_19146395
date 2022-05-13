import pandas as pd
import numpy as np
import cv2
from matplotlib import  pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
from keras.models import load_model
from array import array

train_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory("/Hitori/train",target_size=(150,150),batch_size = 5,class_mode = "categorical")
test_dataset = ImageDataGenerator(rescale=1/255).flow_from_directory("/Hitori/test/",target_size=(150,150),batch_size = 5,class_mode = "categorical")

model  = Sequential()
model.add(Flatten())
model.add(Dense(512, activation = 'relu', input_shape = (150,150,3)))
model.add(Dense(32,activation='relu'))
model.add(Dense(2))

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(train_dataset,epochs = 10,validation_data=test_dataset)



