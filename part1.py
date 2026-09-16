#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

columns = ["CIC0", "SM1_Dz_Z", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]

dataset = "https://raw.githubusercontent.com/Rubenokay/MLAssignment1/master/data.csv"
data = pd.read_csv(dataset, sep=";", header=None, names=columns)

data = data.dropna()
data = data.drop_duplicates()

data = data.reset_index(drop=True)

target = "LC50"
X = data.drop(columns=[target]).to_numpy(float)
y = data[target].to_numpy(float)
features = data.drop(columns=[target]).columns.tolist() # all except target

def split(X, y, size=0.25, state=None):
    sampleSize = X.shape[0]
    rand = np.random.RandomState(state)
    num = rand.permutation(sampleSize)

    testNum = int(round(sampleSize*size))
    testI = num[:testNum]
    trainingI = num[testNum:]

    return X[trainingI], X[testI], y[trainingI], y[testI]

trainX, testX, trainY, testY = split(X, y, size=0.25, state=1)
scale = StandardScaler()
trainXScaled = scale.fit_transform(trainX)
testXScaled = scale.transform(testX)

def update_w_and_b(X, y, w, b, alpha):
    rows = X.shape[0]
    yHat = np.dot(X, w) + b
    error = y-yHat

    dr_dw = 0.0
    dr_db = 0.0

    dr_dw = -2*np.dot(X.T, error)
    dr_db = -2*np.sum(error)

    dr_dw = dr_dw/rows
    dr_db = dr_db/rows

    w = w-alpha*dr_dw
    b = b-alpha*dr_db

    return w,b

def loss(X, y, w, b):
    yHat = np.dot(X, w) + b

    return np.mean((y-yHat) ** 2)

def train(X, y, w, b, alpha, epochs, log_every=None):
    mse = []

    for i in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        lossNow = loss(X, y, w, b)
        mse.append(lossNow)

        if log_every and (i == 0 or i % log_every == 0):
            print("epoch:", i, "loss:", lossNow)

    return w, b, mse

numFeatures = trainXScaled.shape[1]
w = np.zeros(numFeatures)
b = 0.0
alpha = 0.01
epochs = 1000

w2, b2, mse = train(trainXScaled, trainY, w, b, alpha, epochs, log_every=100)

print("Weights\n", w2)
print("\nBias\n", b2)
print("\nTraining Loss\n", loss(trainXScaled, trainY, w2, b2))
print("\nLoss\n", loss(testXScaled, testY, w2, b2))
