import math
import numpy as np
import matplotlib.pyplot as plt

from RandomForest import RandomForest
from DecisionTree import Tree


number_of_points = 100 #number of data points per class
number_of_classes = 3 #number of classes in dataset

#data generation: creates spiral dataset with 4 classes and 100 samples each
def generateData(number_of_points, number_of_classes):

    data = np.empty([3, number_of_classes * number_of_points])

    for i in range(0, number_of_classes):
        data[0, i*number_of_points : (i+1) * number_of_points] = np.float(i) #np.matrix(np.ones(numberOfPoints));
        radius = np.linspace(0.05, 0.9, number_of_points)
        theta  = np.linspace(i*2*math.pi/number_of_classes, \
            i*2*math.pi/number_of_classes + 3*math.pi/2, number_of_points) +\
            np.random.normal(0, .1, number_of_points)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        datum = np.matrix(np.transpose(np.column_stack((x, y))))
        data[1:, i*number_of_points:(i+1)*number_of_points] = datum

    return data

def display(data):
    display_decision_boundary(None)
    display_training_data(data)
    plt.show()

#displays training data for classification
def display_training_data(data):
    colors = ['red', 'green', 'blue', 'yellow', 'orange']

    for i in range(0, number_of_classes):
        plt.scatter(data[1, i*number_of_points:(i+1)*number_of_points], data[2, i*number_of_points:(i+1)*number_of_points], c=colors[i], s=40)

def display_decision_boundary(hists, size):
    plt.imshow(c, interpolation='nearest', extent=[-1,1,-1,1])

#returns histograms in range -1,1 -1,1
def train_random_forest(data, size):
    return RandomForest(data, size, 7, number_of_classes)



    # return rf.create_random_forest(data, 100, 7)
    # m = np.linspace(-1, 1, size)
    # n = np.linsapce(-1, 1, size)
    #
    # histograms = np.empty([size * size])
    #
    # for i in range(size):
    #     for j in range(size):
    #         histograms[i * size + j] = rf.traceTree(np.transpose(np.matrix([m[i], n[j]])))
    #
    # histograms = histograms.reshape((size, size))
    #
    # display_decision_boundary(histograms)

#creates a decision boundary represented as a 1000 x 1000 x 3 matrixå
def create_decision_boundary(forest):
    return None

if __name__ == '__main__':
    data = generateData(number_of_points, number_of_classes)

    forest = train_random_forest(data, 100)

    hist = forest.test_point(np.transpose(np.matrix([0, 0])))

    print(hist)

    # create_decision_boundary(forest)


    #Tree(data, 7, 3)

    #display(data)
