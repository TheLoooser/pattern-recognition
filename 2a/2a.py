# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:27:29 2020

@author: sebas
"""
import sklearn.svm as svm
import sklearn.model_selection as model_selection
import numpy as np
import dataPreparation as prep
import warnings

def warning(*args, **kwargs):
    pass
warnings.warn = warning

train = prep.csvImport('mnist_train.csv')
test = prep.csvImport('mnist_test.csv')

#Deletes every columns where for all rows, the value is 0.
data = prep.dataDeletion(train, test)

train = np.array(data[0])
test = np.array(data[1])

#Uncomment these lines to test it with only a tenth of the data
#train = train[0:round(len(train)/10)]
#test = test[0:round(len(test)/10)]

print(len(train))
print(len(test))

#Splits the variables and the class
xtrain = train[:, 1 : len(train)-1]
ytrain = train[:, 0]

xtest = test[:, 1 : len(train)-1]
ytest = test[:, 0]

#Initiates an SVM classifier with a rbf kernel
rbf = svm.SVC()
##Prepares a cross-validation model
kfold = model_selection.KFold(10)
##Best results during the best paramater search process will be stored in this variable : [accuracy, c, gamma]
bestResult = [0,0,0]
gamma = 1 / (len(xtrain[0]) * xtrain.var());

#Gridsearch, 5 different c and 5 different gamma
for i in range(-2, 3):
    c = 2**i
    rbf.C = c
    for j in range(-2,3):
        rbf.gamma = gamma*(2**j)
        result = np.mean(model_selection.cross_val_score(rbf, xtrain, y = ytrain , cv=kfold  ,n_jobs=-1))
        print('C: ', rbf.C,'  Gamma: ', rbf.gamma, 'Result: ', result)
        if (result > bestResult[0]):
            bestResult = [result, rbf.C, rbf.gamma]
print('Accuracy in train with 10-fold CV: ',bestResult)

rbf.C = bestResult[1]
rbf.gamma = bestResult[2]
#Use of the best C and the best gamma to fit the model
rbf.fit(xtrain, ytrain)
#Predicts the test and returns the accuracy
print(rbf.score(xtest, ytest))


#Initiates an SVM with a linear kernel
lin = svm.LinearSVC()
#Prepares a 10-fold cross-validation model
kfold = model_selection.KFold(10)

#Algorithm to find the best C
#Defines a range, C will be searched between this range
c1 = 0.1
c2 = 30

#Initializes an array containing the different pair of C and accuracy and the number of iteration in the algorithm
results = []
iteration = 5

#Computes the accuracies of the two border Cs
lin.C = c1
result = np.mean(model_selection.cross_val_score(lin, xtrain, y = ytrain, cv=kfold, n_jobs = -1))
results.append([lin.C, result])
lin.C = c2
result = np.mean(model_selection.cross_val_score(lin, xtrain, y = ytrain, cv=kfold, n_jobs = -1))
results.append([lin.C, result])

place = 0

#This search algorithm computes the accuracy of a certain number of C between two border values.
#The C with the best accuracy and his best neighbour tested become the new border values.
#This is done for x iteration defined earlier. It can find a local optimum, doesn't guarantee the global optimum.
for i in range(0, iteration):
    #Defines the size of steps between each C tested
    step = (c2 - c1) / 5
    tempResult = []
    for j in range(1, 5):
        lin.C = c1 + j * step
        result = np.mean(model_selection.cross_val_score(lin, xtrain, y = ytrain, cv=kfold, n_jobs = -1))
        results.insert((place + j),[lin.C, result])
    maxResult = [0,0]
    for r in range(0, len(results)):
        if results[r][1] > maxResult[1]:
            maxResult = [r, results[r][1]]
    if results[maxResult[0]+1][1] > results[maxResult[0]-1][1]:
        place = maxResult[0]
        c1 = results[place][0]
        c2 = results[place+1][0]
    if results[maxResult[0]+1][1] < results[maxResult[0]-1][1]:
        place = maxResult[0]-1
        c1 = results[place][0]
        c2 = results[place+1][0]
    if results[maxResult[0]+1][1] == results[maxResult[0]-1][1]:
        place = maxResult[0]-1
        c1 = results[place-1][0]
        c2 = results[place+1][0]

maxResult = [0,0]

#Search for the best accuracy and its parameter C found during the search algorithm.
for b in range(0, len(results)):
    if results[b][1] > maxResult[1]:
        maxResult = [results[b][0], results[b][1]]
lin.C = maxResult[0]
print(results)

#Fits the train data with the use of the best C
lin.fit(xtrain,ytrain)

#Predicts the test data and returns the accuracy
print(lin.score(xtest, ytest))




