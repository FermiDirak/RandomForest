import math
import numpy as np
import Tree from DecisionTree

#data and label length should be the same
def create_random_forest(data, number_of_trees, min_depth):
    number_of_classes = data[0, :].max() + 1

    forest = np.empty([number_of_trees])

    #create an ensamble of trees
    for t in range(0, number_of_trees):
        #pick sqrt(n) datapoints. Can be redundant datapoints
        pick_count = np.floor(np.sqrt(data.size.m))

        #subset of the dataset to create a tree out of
        subset = np.empty([data.size.n, pick_count])
        for i in range(0, pick_count):
            subset[0, i] = data[:, np.floor(data.size.m * np.random.rand())]

        #create tree
        tree = new DecisionTree(subset, min_depth, number_of_classes);
        forest[t] = tree

    return forest

#classifies a 2 featured instance
def testPoint(forest, instance):
    histograms = np.empty([number_of_classes])

    #go down the tree and find histogram of the point for the tree
    for t in range(0, forest.size.m):
        tree = forest[t]
        histograms[i] = tree.traceNode(instance)

    return np.cumsum(histograms) / number_of_classes
