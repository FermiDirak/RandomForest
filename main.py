import math
import numpy as np
import matplotlib.pyplot as plt

from RandomForest import RandomForest
from DecisionTree import Tree
from NN import NN


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

def display(data, hists):
    display_decision_boundary(hists)
    display_training_data(data)
    plt.show()

#displays training data for classification
def display_training_data(data):
    colors = ['green', 'blue', 'red', 'yellow', 'orange']

    for i in range(0, number_of_classes):
        plt.scatter(data[1, i*number_of_points:(i+1)*number_of_points], data[2, i*number_of_points:(i+1)*number_of_points], c=colors[i], s=40)

def display_decision_boundary(hists):
    plt.imshow(hists, interpolation='nearest', extent=[-1,1,-1,1])

#returns histograms in range -1,1 -1,1
def train_random_forest(data, size):
    return RandomForest(data, size, 7, number_of_classes)

#creates a decision boundary represented as a 1000 x 1000 x 3 matrix
def create_decision_boundary(forest, size):
    def scale_to_grid(i, size):
        return -1 + 2 * (i / size)

    hists = np.zeros([size, size, 3])

    for i in range(0, size):
        for j in range(0, size):
            hists[i, j] = forest.test_point(np.transpose(np.matrix([scale_to_grid(i, size), scale_to_grid(j, size)])))

    return hists

def train_nn(data):
    # print(data.T, np.shape(data))
     print(data.T[:, 1:3].shape)
     # print(data.T[range(400), 0].shape)

     nn = NN(data.T[: ,1:], data.T[:, 0])

     nn.train()
     nn.display()

if __name__ == '__main__':
    data = generateData(number_of_points, number_of_classes)
    # testing if master different now! w
    # train_softmax(data)
    # train_nn(data)

    print('creating forest')
    forest = train_random_forest(data, 200)
    print('forest created')

    print('creating decison boundary')
    hists = create_decision_boundary(forest, 25)
    print('decision boundary created')

    print('displaying data and decision boundary')
    display(data, hists)
