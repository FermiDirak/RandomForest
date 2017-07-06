import math
import numpy as np

#data and label length should be the same
def RandomForest(data, labels, number_of_trees, min_depth):
    number_of_classes = labels.max()
    data = [labels; data]

    forest = np.empty([0, number_of_trees])

    #create a bunch of trees and stuff
    for t in range(0, number_of_trees):
        #pick sqrt(n) datapoints. Can be redundant datapoints
        pick_count = np.floor(np.sqrt(data.size.m))

        #subset of the dataset to create a tree out of
        subset = np.empty([data.size.n, pick_count])
        for i in range(0, pick_count):
            subset[0, i] = data[:, np.floor(data.size.m * np.random.rand())]

        #create tree
        tree = new DecisionTree(subset);
        forest[0, t] = tree

    forest = np.matrix(forest)
    return forest

#classifies a 2 featured new instance
def testPoint(forest, new_instance):
    histograms = np.empty([1, number_of_classes])

    #go down the tree and find histogram of the point for the tree
    for t in range(0, forest.size.m)
        #@TODO find the histogram for a given point for the tree t

    return 0
