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

def costMatrix (m1, m2, cn, ce):
    matrix = np.empty([len(m1),])
    return matrix

# transform list of molecule nodes to dictionnary (key: molecule_id, value: molecule_symbols)
def moleculeNodeListToDict(nodesList):
    nodesDict = dict()
#     id_counter = 0
    for node in nodesList:
        nodesDict[node[0]] = node[1]
#         nodesDict[id_counter] = {"id": node[0], "symbols": node[1]}
#         id_counter += 1
    return nodesDict

# transform list of molecule edges to dictionnary (key: molecule_id, value: list of molecule_edges)
def moleculeEdgeListToDict(edgesList):
    edgesDict = dict()
    for edge in edgesList:
        edgesDict[edge[0]] = edge[1]
    return edgesDict

#nr: lenght of molecule
def moleculeEdgesToCount(nr, edgesList):
    count = list()
    for i in range(nr):
        list.append(sum(x.count(i) for x in edgesList))
    return count

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    