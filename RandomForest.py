import math
import numpy as np

#data and label length should be the same
def RandomForest(data, labels, numberOfTrees, minDepth):
    numberOfClasses = labels.max()
    data = [labels; data]

    forest = np.empty([0, numberOfTrees])

    #create a bunch of trees and stuff
    for t in range(0, numberOfTrees):
        #pick sqrt(n) datapoints. Can be redundant datapoints
        pickCount = np.floor(np.sqrt(data.size.m))

        #subset of the dataset to create a tree out of
        subset = np.empty([data.size.n, pickCount])
        for i in range(0, pickCount):
            subset[0, i] = data[:, np.floor(data.size.m * np.random.rand())]

        #create tree
        tree = new DecisionTree(subset);
        forest[0, t] = tree

    forest = np.matrix(forest)
    return forest

#classifies a 2 featured new instance
def testPoint(forest, newInstance):
    for t in range(0, forest.size.m)
