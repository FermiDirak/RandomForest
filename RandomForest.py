import math
import numpy as np

from DecisionTree import Tree

class RandomForest:
    def __init__(self, data, number_of_trees, min_depth, number_of_classes):
        self.data = data
        self.number_of_trees = number_of_trees
        self.min_depth = min_depth
        self.number_of_classes = number_of_classes

        self.forest = self.create_random_forest()


    #data and label length should be the same
    def create_random_forest(self):
        number_of_classes = self.data[0, :].max() + 1

        forest = []

        #create an ensamble of trees
        for t in range(0, self.number_of_trees):
            #pick sqrt(n) datapoints. Can be redundant datapoints
            # no idea what u were trying to do with size attribute
            pick_count = int(math.floor(np.sqrt(self.data.shape[1])))

            #subset of the dataset to create a tree out of
            subset = np.empty([self.data.shape[0], pick_count])

            #populating the subset
            for i in range(0, pick_count):
                sample = self.data[:, int(np.floor(self.data.shape[1] * np.random.rand()))]
                subset[:, i] = sample

            #create tree
            tree = Tree(subset, self.min_depth, number_of_classes)
            forest.append(tree)

        return forest

    #classifies a 2 featured instance
    def test_point(self, instance):
        histograms = np.empty([self.number_of_classes])

        #go down the tree and find histogram of the point for the tree
        for t in range(0, len(self.forest)):
            tree = self.forest[t]
            histograms[i] = tree.trace_tree(instance)

        return np.sum(histograms, axis = 0) / self.number_of_classes
