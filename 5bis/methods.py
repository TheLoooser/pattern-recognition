import os
import xml.etree.ElementTree as ET
import numpy as np

def importMolecule(path):
    roots = []
    for molecule in os.listdir(path):
        roots.append([int(molecule.replace('.gxl','')), ET.parse(path + '\\' + molecule).getroot()])
    return roots



def moleculeToList(molecules):
    nodesList = []
    edgesList = []
    for molecule in molecules:
        nodes = [];
        edges = [];
        for node in molecule[1].iter('node'):
            for attribute in node.getchildren():
                if (attribute.attrib['name'] == 'symbol'):
                    for string in attribute.getchildren():
                        nodes.append(string.text.replace(' ', ''))
        for element in molecule[1].iter('edge'):
            edges.append([int(element.attrib['from'].replace('_', '')), int(element.attrib['to'].replace('_', ''))])
        nodesList.append([molecule[0], nodes])
        edgesList.append([molecule[0], edges])
    return [nodesList, edgesList]



def costMatrix (n1, e1, n2, e2, cn = 1, ce = 0.1):
    #initialize the matrix
    matrix = np.zeros([len(n1)+len(n2),len(n1)+len(n2)])
    

    for i in range(0, len(n1)):
        #top-left part (substition)
        for j in range(0, len(n2)):
            if (n1[i] == n2[j]):
                matrix.itemset((i,j),0)
            else:
                matrix.itemset((i,j),2*cn)
        edges = 0
        
        #top-right part (deletions)
        for edge in e1:
            if (edge[0] or edge[1]) == i:
                edges = edges + 1
        for j in range(len(n2), len(n2) + len(n1)):
            if (j - len(n2) == i):
                matrix.itemset((i, j), cn + edges * ce)
            else:
                matrix.itemset((i,j), np.inf)    
    
    #bottom-left part (insertion)
    for i in range(len(n1), len(n1) + len(n2)):
        edges = 0
        for edge in e2:
            if (edge[0] or edge[1]) == j:
                edges = edges + 1
        for j in range(0, len(n2)):
            if (j + len(n1) == i):
                matrix.itemset((i, j), cn + edges * ce)
            else:
                matrix.itemset((i,j), np.inf)        
    return matrix


#knnsearch with return of accuracy
def knnsearch(k, validData, trainData, distanceMatrix):
    #initialize the result 2d list with the name (number) of the validationData for every element
    nearestNeighbours = []
    for i in range(0, len(validData)):
        nearestNeighbours.append([validData[i][0]])
    
    i = 0
    while i < len(validData):
        j = 0
        retained = []
        biggestRetainedDistance = 0
        while j < len(trainData):
            value = distanceMatrix[j][i] 
            #Fills an array with k element with the first distances calculated
            if (j < k):
                retained.append([value, trainData[j][1]])
                # The biggest value of k-elements array is stored to be compared with the new distances calculated
                # It avoids the need to store every distances calculated, if they are higher than this value,
                # they are ignored.
                if (j == 0):
                    biggestRetainedDistance = value
                else:
                    if value > biggestRetainedDistance:
                        biggestRetainedDistance = value
            else:
                if (j == k):
                    retained.sort()
    
                if value < biggestRetainedDistance:
                    m = 0
                    while value > retained[m][0]:
                        m = m + 1
                    retained.insert(m,[value,trainData[j][1]])
                    del retained[-1]
                    biggestRetainedDistance = retained[k-1][0]    
                    
            j = j + 1
        for j in range(0,len(retained)):
            nearestNeighbours[i].append(retained[j][1])
        i = i + 1
    classes = []

    for i in range(0, len(nearestNeighbours)):
        aCounter = 0
        iCounter = 0
        for j in range(1, len(nearestNeighbours[i])):
            if (nearestNeighbours[i][j]) == 'a':
                aCounter = aCounter + 1
            if (nearestNeighbours[i][j]) == 'i':
                iCounter = iCounter + 1
        if aCounter < iCounter:
            classes.append([nearestNeighbours[i][0], 'i'])
        if iCounter < aCounter:
            classes.append([nearestNeighbours[i][0], 'a'])
        
    trueCounter = 0;
    for j in range(0, len(validData)):
        if classes[j][1] == validData[j][1] :
            trueCounter = trueCounter + 1
    accuracy = trueCounter / len(validData)
    return accuracy
    