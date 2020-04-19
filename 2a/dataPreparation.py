# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:27:58 2020

@author: sebas
"""
#Imports a csv-file named path, loads in a list and ensures that the output datas are integers
def csvImport(path):
    import csv
    with open(path, newline='') as train:
        reader = csv.reader(train)
        stringTrainRows = []
        for row in reader:
            stringTrainRows.append(row)
    trainRows = []
    for row in stringTrainRows:
        trainRows.append([int(i) for i in row if i!=0])
    return trainRows


# Deletes every columns where, for each rows, the value is 0.
def dataDeletion(traindata, testdata):
    columnsDel = []
    for i in range(1, len(traindata[0])):
        sumTrain = 0
        sumTest = 0
        for j in range(0, len(traindata)):
            sumTrain = sumTrain + traindata[j][i]
        for j in range(0, len(testdata)):
            sumTest = sumTest + testdata[j][i]
        if (sumTrain + sumTest == 0):
            columnsDel.insert(0, i)
    for i in columnsDel:
        for k in range(0, len(traindata)):
            del traindata[k][i]
        for k in range(0, len(testdata)):
            del testdata[k][i]
    return [traindata, testdata]

