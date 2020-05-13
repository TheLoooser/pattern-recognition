import numpy as np
from matplotlib import pyplot as plt
import PIL
import cv2
import time
import pickle
import methods as mt

trees = []
roots = []

nodesList = [[]]
edgesList = [[]]

roots = mt.importMolecule('gxl')


listedMolecules = mt.moleculeToList(roots)
nodesList = listedMolecules[0]
edgesList = listedMolecules[1]
print(nodesList)
print(edgesList)

