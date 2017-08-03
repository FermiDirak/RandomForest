import math
import numpy as np

class Node:

    def __init__(self, data, depth, labels_count):
        self.data = data
        self.depth = depth
        self.labels_count = int(labels_count)

        self.split = self.get_best_gini_split()

        self.left = None
        self.right = None

        if (depth >= 1):
            self.add_node('left')
            self.add_node('right')

    def add_node(self, direction):
        if self.depth <= 0:
            return

        if direction == 'left':
            left_data = self.get_left_data(self.data, self.split)

            if left_data is not None and left_data.shape[1] != 0:
                self.left = Node(left_data, self.depth - 1, self.labels_count)

        elif direction == 'right':
            right_data = self.get_right_data(self.data, self.split)

            if right_data is not None and right_data.shape[1] != 0:
                self.right = Node(right_data, self.depth - 1, self.labels_count)

        else:
            error('add_node took in an improper direction')

    #gets a random split point for the dataset
    def get_random_split(self, dataset):
        split = np.transpose(np.matrix(np.zeros(2)))

        if dataset.shape[1] <= 1:
            return split

        coordN = int(np.round(np.random.rand()))
        coordM = int(np.floor(dataset.shape[1] * np.random.rand()) - 1)

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
        dataset = np.matrix(dataset)

        if dataset.shape[1] == 0:
            return None

        #create an empty matrix the size of the dataset
        split_dataset = np.matrix(np.zeros([int(dataset.shape[0]), int(dataset.shape[1])]))

        feature_index = 0
        if (split[1, 0] != 0):
            feature_index = 1
        split_value = split[feature_index, 0]

        j = 0
        for i in range(0, dataset.shape[1]):
            feature_value = dataset[feature_index + 1, i]

            if ((direction == 'left' and feature_value <= split_value) or (direction == 'right' and feature_value > split_value)):

                split_dataset[:, j] = dataset[:, i]
                j += 1

        split_dataset = split_dataset[:,0:j]

        if (split_dataset.shape[1] == 0):
            return None
        else:
            return split_dataset

    #get the best split vector for dataset using Gini impurity
    def get_best_gini_split(self):
        best_split = np.transpose(np.matrix(np.zeros(2)))
        best_gini = 2

        for i in range(0, int(self.data.shape[0] - 1)):
            for j in range(0, int(self.data.shape[1])):
                split = np.transpose(np.matrix(np.zeros(2)))
                split[i, 0] = self.data[i+1, j]

                left_split = self.get_left_data(self.data, split)
                right_split = self.get_right_data(self.data, split)

                left_hist = Node.calc_histogram(left_split, self.labels_count)
                right_hist = Node.calc_histogram(right_split, self.labels_count)

                gini_left = Node.calc_gini(left_hist, self.labels_count)
                gini_right = Node.calc_gini(right_hist, self.labels_count)

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

        if dataset is None:
            return histogram

        number_of_datum = dataset.shape[1]

        for i in range(number_of_datum):
            label_id = dataset[0, i]

            histogram[int(label_id)] += 1

        histogram = histogram / number_of_datum

        return histogram

    #instance is a feature vector. Compares to the Tree
    @staticmethod
    def trace_node(node, instance):
        split_index = 0
        if (abs(node.split[0, 0]) <= 0.000001):
            split_index = 1

        #if at leaf, we return the histogram
        if (node.left == None) and (node.right == None):
            return Node.calc_histogram(node.data, node.labels_count)

        if (instance[split_index, 0] <= node.split[split_index, 0] and node.left != None):
            return Node.trace_node(node.left, instance)
        elif (node.right != None):
            return Node.trace_node(node.right, instance)
        else:
            return Node.trace_node(node.left, instance)

class Tree:
    def __init__(self, dataset, min_depth, labels_count):
        self.tree = None
        self.min_depth = min_depth
        self.labels_count = labels_count
        self.dataset = dataset

        self.tree = Node(dataset, min_depth, labels_count)

    def trace_tree(self, instance):
        return Node.trace_node(self.tree, instance)
