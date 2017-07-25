import math
import numpy as np

class Node:
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth

        self.split = self.getRandomSplit(self.data)

        if (depth > 0):
            self.left = Node(self.get_left_split(self.data, self.split), self.depth - 1))
            self.right = Node(self.get_right_split(self.data, self.split), self.depth - 1))
        else (depth <= 0):
            self.left = None
            self.right = None


    def traceNode(self, instance):
        splitIndex = 0
        if (self.split[0, 0] == 0):
            splitIndex = 1

        if (self.left == None) or (self.right == None):
            return calc_histogram(self.data, self.labels_count)

        if (instance[splitIndex, 0] <= self.split[splitIndex, 0]):
            return(traceNode(self.left))
        else:
            return(traceNode(self.right))

    #gets a random split point for the dataset
    def getRandomSplit(self, dataset):
        split = np.transpose(np.matrix(np.zeros(2)))
        coordN = int(np.round(np.random.rand()))
        coordM = int(np.floor(dataset.shape[1] * np.random.rand()))

        split[coordN, 0] = dataset[coordN + 1, coordM]

        return split

    #gets split where top right are 'right' and bottom left are 'left'. returns left subset of data from split
    @staticmethod
    def get_left_split(dataset, split):
        return Node.get_split(dataset, split, 'left')

    #gets split where top right are 'right' and bottom left are 'left'. returns right subset of data from split
    @staticmethod
    def get_right_split(dataset, split):
        return Node.get_split(dataset, split, 'right')

    #returns split. pass in 'left' for left and 'right' for right for direction to get that split
    @staticmethod
    def get_split(dataset, split, direction):
        split_dataset = np.empty([dataset.shape[0], dataset.shape[1]])
        is_x_split = (split[1,0] == 0)
        feature = 0
        if not is_x_split:
            feature = 1
        split_value = split[feature, 0]

        j = 0
        for i in range(0, dataset.shape[1]):
            current_instance = dataset[feature, i]

            if ((direction == 'left' and current_instance[feature, 0] <= split_value) or (direction == 'right' and current_instance[feature, 0] > split_value)):
                split_dataset[:, j] = dataset[:, i]
                j += 1

        split_dataset = split_dataset[:,0:j+1]
        return split_dataset

    #get the best split vector for dataset using Gini impurity
    @staticmethod
    def getBestGiniSplit(dataset, labels_count):
        best_split = np.transpose(np.matrix(np.zeros(2)))
        best_gini = 2

        for i in range(0, int(dataset.shape[0] - 1)):
            for j in range(0, int(dataset.shape[1])):
                split = np.transpose(np.matrix(np.zeros(2)))
                split[i, 0] = dataset[i+1, j]

                gini_left = calc_gini(calc_histogram(get_left_split(dataset, split), labels_count))
                gini_right = calc_gini(calc_histogram(get_right_split(dataset, split), labels_count))

                gini = gini_left + gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_split = split

        return best_split

    #calculates gini value of a given histogram
    @staticmethod
    def calc_gini(histogram, labels_count):
        gini = 0

        for i in range(0, labels_count):
            gini += histogram[i] * histogram[i]
        gini = 1 - gini
        return gini

    #returns a 1 x labelCount histogram representation of the data
    @staticmethod
    def calc_histogram(dataset, labels_count):
        histogram = np.zeros(labels_count)
        number_of_datum = datset.shape[1]

        for i in range(number_of_datum):
            label_id = dataset[0, i]
            histogram[label_id] += 1

        histogram = histogram / number_of_datum
        return histogram

class Tree:
    def __init__(self, dataset, min_depth, labels_count):
        self.tree = None
        self.min_depth = min_depth
        self.labels_count = labels_count
        self.dataset = dataset

        self.tree = Node(dataset, min_depth)

    def traceTree(self, instance):
        return self.tree.traceNode(instance)
