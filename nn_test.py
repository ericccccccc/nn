# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:22:20 2020

@author: Eric
"""

import numpy as np 
from scipy.stats import norm
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import matplotlib.pyplot as plt


def crtCDF(x):
    if(type(x) == np.ndarray):
        loc = x.mean()
        scale = x.std()
        N = x.size
        pos = norm.cdf(x, loc, scale)*N
        return pos
    else:
        print("Wrong Type! x must be np.ndarray ~")    
        return

x = np.random.random_integers(1000,size=100)
x = np.sort(x)
y = crtCDF(x)
norm_x = preprocessing.scale(x)    # 標準化: 零均值化


model  = Sequential()
model.add(Dense(32, input_dim=1, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

sgd=keras.optimizers.SGD(lr=0.0001)    # lr:學習率,可調參數
model.compile(loss="mse", optimizer=sgd, metrics=["mse"])
model.fit(norm_x, y, epochs=1000, batch_size=32, verbose=0)  # norm_x:訓練資料, y:訓練目標
pred_y = model.predict(norm_x)

plt.title("Neural Network Model")
plt.plot(x, y, '.',label="Origin")
plt.plot(x, pred_y,'.',label="Model")
plt.legend()
plt.xlabel("Key")
plt.ylabel("Pred_Pos = CDF(Key)")
plt.show()