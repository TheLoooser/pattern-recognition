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