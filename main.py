import math
import numpy as np
import matplotlib.pyplot as plt

from NN import Softmax, NN

import RandomForest as rf


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
    display_decision_boundary(None, None, None)
    display_training_data(data)
    plt.show()

#displays training data for classification
def display_training_data(data):
    colors = ['red', 'green', 'blue', 'yellow', 'orange']

    for i in range(0, number_of_classes):
        plt.scatter(data[1, i*number_of_points:(i+1)*number_of_points], data[2, i*number_of_points:(i+1)*number_of_points], c=colors[i], s=40)

def display_decision_boundary(hists):
    nx = 100
    ny = 100

    r = np.random.random(ny * nx).reshape((ny, nx))
    g = np.random.random(ny * nx).reshape((ny, nx))
    b = np.random.random(ny * nx).reshape((ny, nx))

    c = np.dstack([r,g,b])

    plt.imshow(c, interpolation='nearest', extent=[-1,1,-1,1])


def train_softmax(data):
    print(data.T, np.shape(data))
    print(data.T[:, 1:3].shape)
    print(data.T[:, 0].shape)
    softmax = Softmax(data.T[: ,1:3], data.T[:, 0])
    softmax.train()

def train_nn(data):
    # print(data.T, np.shape(data))
    print(data.T[:, 1:3].shape)
    # print(data.T[range(400), 0].shape)

    nn = NN(data.T[: ,1:], data.T[:, 0])

    nn.train()
    nn.display()

#returns histograms in range -1,1 -1,1
def train_random_forest(data):
    forest = rf.create_random_forest(data, 100, 7)
    

if __name__ == '__main__':
    data = generateData(number_of_points, number_of_classes)
    # train_softmax(data)
    # train_nn(data)

    train_random_forest(data)

    display(data)
