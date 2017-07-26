import math
import numpy as np

class Node:
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth

        self.split = self.get_random_split(self.data)

        if (depth > 0):
            self.left = Node(self.get_left_data(self.data, self.split), self.depth - 1)
            self.right = Node(self.get_right_data(self.data, self.split), self.depth - 1)
        else:
            self.left = None
            self.right = None

    def trace_node(self, instance):
        split_index = 0
        if (self.split[0, 0] == 0):
            split_index = 1

        #if at leaf, we return the histogram
        if (self.left == None) or (self.right == None):
            return calc_histogram(self.data, self.labels_count)

        if (instance[split_index, 0] <= self.split[split_index, 0]):
            return(self.trace_node(self.left))
        else:
            return(self.trace_node(self.right))

    #gets a random split point for the dataset
    def get_random_split(self, dataset):
        split = np.transpose(np.matrix(np.zeros(2)))
        coordN = int(np.round(np.random.rand()))
        coordM = int(np.floor(dataset.shape[1] * np.random.rand()))

        split[coordN, 0] = dataset[coordN + 1, coordM]

        return split

    #gets split where top right are 'right' and bottom left are 'left'. returns left subset of data from split
    def get_left_data(self, dataset, split):
        return self.get_split_data(dataset, split, 'left')

    #gets split where top right are 'right' and bottom left are 'left'. returns right subset of data from split
    def get_right_data(self, dataset, split):
        return self.get_split_data(dataset, split, 'right')

    #returns split. pass in 'left' for left and 'right' for right for direction to get that split
    def get_split_data(self, dataset, split, direction):
        #create an empty matrix the size of the dataset
        split_dataset = np.empty([dataset.shape[0], dataset.shape[1]])
        is_x_split = (split[1,0] == 0)
        feature = 0
        if not is_x_split:
            feature = 1
        split_value = split[feature, 0]

        j = 0
        for i in range(0, dataset.shape[1]):
            current_instance = dataset[feature, i]

            if ((direction == 'left' and current_instance <= split_value) or (direction == 'right' and current_instance > split_value)):
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

    def trace_tree(self, instance):
        return self.tree.trace_node(instance)
