import math
import numpy as np

from NN import swag #imports

#data and label length should be the same
def RandomForest(data, labels, numberOfTrees, minDepth):
    numberOfClasses = labels.max()
    data = [labels; data]

    #create a bunch of trees and stuff
    for t in range(0, numberOfTrees):
        #pick sqrt(n) datapoints. Can be redundant datapoints
        pickCount = np.floor(np.sqrt(data.size.m))

        #subset of the dataset to create a tree out of
        subset = np.empty([data.size.n, pickCount])
        for i in range(0, pickCount):
            subset[0, i] = data[:, np.floor(data.size.m * np.random.rand())]

        #create tree



#get the best split point for dataset
def getRandomSplit(dataset):
    split = np.transpose(np.matrix(np.zeros([0, 0])))
    coordN = np.round(np.random.rand())
    coordM = np.floor(dataset.size.m * np.random.rand())

    split[coordN, 0] = dataset[coordN + 1, coordM]

    return split


#calculates gini value of a given dataset
def calcGini(histogram, labelsCount):
    gini = 0
    for i in range(0, histogram.size.m)
        gini += histogram[0, i] * histogram[0, i]
    gini = 1 - gini
    return gini

def getHistogram(dataset, labelsCount):
    labels = np.empty([labelsCount, 0])
    for i in range(data.size.m)

if __name__ == '__main__':
    swag()
