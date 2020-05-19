import numpy as np
import methods as mt
import csv
from scipy.optimize import linear_sum_assignment


with open('train.txt', newline='') as train:
   reader = csv.reader(train, delimiter=' ')
   stringTrainRows = []
   for row in reader:
        stringTrainRows.append(row)
trainData = []
for row in stringTrainRows:
    trainData.append([int(row[0]),row[1]])


with open('valid.txt', newline='') as train:
   reader = csv.reader(train, delimiter=' ')
   stringValidRows = []
   for row in reader:
        stringValidRows.append(row)
validData = []
for row in stringValidRows:
    validData.append([int(row[0]),row[1]])
roots = []
roots = mt.importMolecule('gxl')
trainNodes = []
validNodes = []
trainEdges = []
validEdges = []
listedMolecules = mt.moleculeToList(roots)
nodesList = listedMolecules[0]
edgesList = listedMolecules[1]
print(len(nodesList))
for i in range(0, len(nodesList)):
    for row in trainData:
        if (row[0] == nodesList[i][0]):
            trainNodes.append(nodesList[i])
            trainEdges.append(edgesList[i])
    for row in validData:
        if (row[0] == nodesList[i][0]):
            validNodes.append(nodesList[i])
            validEdges.append(nodesList[i])
distanceMatrix = np.zeros((len(trainNodes), len(validNodes)))
for i in range(0, len(trainNodes)):
    for j in range(0, len(validNodes)):
        costMatrix = mt.costMatrix(trainNodes[i][1], trainEdges[i][1], validNodes[j][1], validEdges[j][1])
        distanceMatrix.itemset((i,j), costMatrix[linear_sum_assignment(costMatrix)].sum())
print(distanceMatrix)
accuracy = mt.knnsearch(5, validData, trainData, distanceMatrix)
print(accuracy)





