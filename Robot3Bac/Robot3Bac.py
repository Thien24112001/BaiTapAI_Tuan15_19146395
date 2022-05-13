import pandas as pd
from matplotlib import  pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from math import cos, sin, pi
from pandas import DataFrame
import numpy as np

theta1 = []
theta2 = []
theta3 = []
px = []
py = []
alpha = []
l1 = 50
l2 = 40
l3 = 20
for t1 in range (0,80):
    for t2 in range (0,170):
        for t3 in range (0,170):
            theta1.append(t1)
            theta2.append(t2)
            theta3.append(t3)
            px.append(round(l1*cos(t1*pi/180) + l2*cos((t1+t2)*pi/180) + l3*cos((t1+t2+t3)*pi/180),2))
            py.append(round(l1*sin(t1*pi/180) + l2*sin((t1+t2)*pi/180) + l3*sin((t1+t2+t3)*pi/180),2))
            alpha.append(t1+t2+t3)
df = DataFrame(np.c_[theta1,theta2,theta3,px,py,alpha],columns = ['theta1','theta2','theta3','px','py','alpha'])
export_csv = df.to_csv (r'3axis_robot_small.csv', index = None, header=True)

data = pd.read_csv('3axis_robot_small.csv')
theta = data.drop(data.columns[3:6],axis=1)
pos_al = data.drop(data.columns[0:3],axis=1)
pos_al.head()

x_train,x_test,y_train,y_test = train_test_split(pos_al,theta,test_size=0.25)
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_train = x_train.data.tolist()

x_test = standard_scaler.fit_transform(x_test)
x_test = x_test.data.tolist()
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Dense(128,kernel_initializer='normal',activation='relu',input_shape=(3,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(3))
model.summary()

model.compile(loss='mse',optimizer=RMSprop(), metrics=['mean_absolute_error'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test)

