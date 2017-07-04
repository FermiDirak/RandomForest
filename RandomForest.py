import math
import numpy as np

#data and label length should be the same
def RandomForest(data, labels, numberOfTrees, minDepth):


    #create a bunch of trees and stuff
    for t in range(0, numberOfTrees):
        #pick sqrt(n) datapoints. Can be redundant datapoints
        pickCount = np.floor(np.sqrt(data.size.m))

        #subset of the dataset to create a tree out of
        subset = []
        for i in range(0, pickCount):
            subset.push(data[:,math.random(data.size.m)])

        #create tree


def getSplit(dataset, nFeatures):
    
