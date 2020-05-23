import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
from munkres import Munkres, print_matrix

def importMolecule(path):
    roots = []
    for molecule in os.listdir(path):

        roots.append([int(molecule.replace('.gxl','')), ET.parse(path + '\\' + molecule).getroot()])
    return roots

def moleculeToList(molecules):
    nodesList = []
    # edgesList = []
    edgesDict = dict()
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
       # edgesList.append([molecule[0], edges])
        edgesDict[molecule[0]] = edges
    
    # convert to dicts
    nodesDict = dict()
    for node in nodesList:
        nodesDict[node[0]] = node[1]
        
    edgesCountDict = dict()
    for molecule in nodesList:
        nr = len(molecule[1])
        edgesCountDict[molecule[0]] = moleculeEdgesListToEdgeCountList(nr, edgesDict[molecule[0]])

    return nodesDict, edgesDict, edgesCountDict

# # transform list of molecule nodes to dictionnary (key: molecule_id, value: molecule_symbols)
# def moleculeNodeListToDict(nodesList):
#     nodesDict = dict()
# #     id_counter = 0
#     for node in nodesList:
#         nodesDict[node[0]] = node[1]
# #         nodesDict[id_counter] = {"id": node[0], "symbols": node[1]}
# #         id_counter += 1
#     return nodesDict

# # transform list of molecule edges to dictionnary (key: molecule_id, value: list of molecule_edges)
# def moleculeEdgeListToDict(edgesList):
#     edgesDict = dict()
#     for edge in edgesList:
#         edgesDict[edge[0]] = edge[1]
#     return edgesDict

# counts the number of edges of each node (of a single molecule)
def moleculeEdgesListToEdgeCountList(nr, edgesList):
    count = list()
    for i in range(nr):
        count.append(sum(x.count(i+1) for x in edgesList))
    return count

# # transforms the list of edges into a dictionnary which counts the number of edges of each node (of ever molecule)
# def moleculeEdgesListToEdgeCountDict(nodesList, edgesDict):
#     edgesCountDict = dict()
#     for molecule in nodesList:
#         nr = len(molecule[1])
#         edgesCountDict[molecule[0]] = moleculeEdgesListToEdgeCountList(nr, edgesDict[molecule[0]])
#     return edgesCountDict


# lecture 10 slide 21 (bipartite graph matching)
def BP(molecule1, molecule2, edges1, edges2, Cn=1, Ce=1):
    #1. build dirac cost matrix
    matrix = np.matrix(np.ones((len(molecule1)+len(molecule2),len(molecule1)+len(molecule2))) * np.inf)
    #Upper left: substitutions
    for i in range(0, len(molecule1)):
        for j in range(0, len(molecule2)):
            if not(molecule1[i] == molecule2[j]):
                val = Ce*abs(edges1[i] - edges2[j]) #number of node deletions/insertions (times the cost)
                matrix.itemset((i,j),2*Cn + val)            
            else:
                val = Ce*abs(edges1[i] - edges2[j]) #number of node deletions/insertions
                matrix.itemset((i,j), val)
    #Upper right: deletions
    for i in range(0, len(molecule1)):
        val = Cn + Ce*edges1[i] #cost of deleting the node plus number of nodes times cost of deleting all edges of this node
        matrix.itemset((i,len(molecule2) +i), val)
    #Lower left: insertions
    for i in range(0, len(molecule2)):
        val = Cn + Ce*edges2[i] #cost of inserting the node plus number of nodes times cost of inserting all edges of this node
        matrix.itemset((len(molecule1)+i,i), val)
    #Lower right: zeros
    for i in range(0, len(molecule2)):
        for j in range(0, len(molecule1)):
            matrix.itemset((len(molecule1) + i,len(molecule2) + j),0) 
            
#     return matrix
    
    #2. find optimal assignment (using Hungarian algorithm)
    #replace infinity with max int
    matrix = np.where(matrix == np.inf, sys.maxsize, matrix)
    #Hungarian algorithm (get total cost)
    matrix = matrix.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    
    #3. calculate edit path distance/cost (of the optimal assignment)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
    
    #4. return distance/cost
    return total
  