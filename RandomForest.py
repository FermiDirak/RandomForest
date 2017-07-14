import math
import numpy as np

from DecisionTree import Tree

#data and label length should be the same
def create_random_forest(data, number_of_trees, min_depth):
    number_of_classes = data[0, :].max() + 1

    forest = []

    #create an ensamble of trees
    for t in range(0, number_of_trees):
        #pick sqrt(n) datapoints. Can be redundant datapoints
        # no idea what u were trying to do with size attribute
        pick_count = int(math.floor(np.sqrt(data.shape[1])))

        #subset of the dataset to create a tree out of
        subset = np.empty([data.shape[0], pick_count])
        for i in range(0, pick_count):
            subset[:, i] = data[:, np.floor(data.shape[1] * np.random.rand())]

        #create tree
        tree = Tree(subset, min_depth, number_of_classes)
        forest.push(tree)

    return forest

#classifies a 2 featured instance
def testPoint(forest, instance):
    histograms = np.empty([number_of_classes])

    #go down the tree and find histogram of the point for the tree
    for t in range(0, forest.shape[1]):
        tree = forest[t]
        histograms[i] = tree.traceNode(instance)

    return np.sum(histograms, axis = 0) / number_of_classes
