#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

columns = ["CIC0", "SM1_Dz_Z", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]

dataset = "https://raw.githubusercontent.com/Rubenokay/MLAssignment1/master/data.csv"
data = pd.read_csv(dataset, sep=";", header=None, names=columns)

print(data.shape)
print(list(data.columns))
print(data.isnull().sum())

data = data.dropna()
data = data.drop_duplicates()

data = data.reset_index(drop=True)
print(data.shape)#same

target = "LC50"
X = data.drop(columns=[target]).to_numpy(float)
y = data[target].to_numpy(float)
features = data.drop(columns=[target]).columns.tolist() # all except target

print(X)
print(y)
print(features)

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

print(trainXScaled.shape)
print(testXScaled.shape)

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


#It is coded with only one variable, you will have to use multiple. Line 40 update wb  line 22 wb learning rate alpha pretty simple.
#Code this algorithm from scratch. Learning rate is the step size and you will have to try various values of the learning grid. LIne 60 fo learning rate.
#TODO source
# def plot_original_data():
#     x, y = np.loadtxt("data.txt", delimiter= "\t", unpack = True)

#     plt.scatter(x, y, color='#1f77b4', marker='o')

#     plt.xlabel("Spendings, M$")
#     plt.ylabel("Sales, Units")
#     plt.title("Sales as a function of radio ad spendings.")
#     #plt.show()
#     fig1 = plt.gcf()
#     fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
#     fig1.savefig('../../Illustrations/gradient_descent-1.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#     fig1.savefig('../../Illustrations/gradient_descent-1.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#     fig1.savefig('../../Illustrations/gradient_descent-1.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)




#     N = len(spendings)

#     for i in range(N):
#         dr_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
#         dr_db += -2 * (sales[i] - (w * spendings[i] + b))

#     # update w and b
#     w = w - (dr_dw/float(N)) * alpha
#     b = b - (dr_db/float(N)) * alpha

#     return w, b

# def train(spendings, sales, w, b, alpha, epochs):
#     image_counter = 2;
#     for e in range(epochs):
#         w, b = update_w_and_b(spendings, sales, w, b, alpha)

#         # log the progress
#         if (e == 0) or (e < 3000 and e % 400 == 0) or (e % 3000 == 0):
#             print("epoch: ", str(e), "loss: "+str(loss(spendings, sales, w, b)))
#             print("w, b: ", w, b)
#             plt.figure(image_counter)
#             axes = plt.gca()
#             axes.set_xlim([0,50])
#             axes.set_ylim([0,30])
#             plt.scatter(spendings, sales)
#             X_plot = np.linspace(0,50,50)
#             plt.plot(X_plot, X_plot*w + b)
#             #plt.show()
#             fig1 = plt.gcf()
#             fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
#             fig1.savefig('../../Illustrations/gradient_descent-' + str(image_counter) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#             fig1.savefig('../../Illustrations/gradient_descent-' + str(image_counter) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#             fig1.savefig('../../Illustrations/gradient_descent-' + str(image_counter) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
#             image_counter += 1
#     return w, b

#     N = len(spendings)
#     total_error = 0.0
#     for i in range(N):
#         total_error += (sales[i] - (w*spendings[i] + b))**2
#     return total_error / N

# x, y = np.loadtxt("data.txt", delimiter= "\t", unpack = True)
# #w, b = train(x, y, 0.0, 0.0, 0.001, 15000)

# plot_original_data()

# def predict(x, w, b):
#     return w*x + b
# x_new = 23.0
# y_new = predict(x_new, w, b)
# print(y_new)
