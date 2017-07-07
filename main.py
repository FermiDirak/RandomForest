import math
import numpy as np
import matplotlib.pyplot as plt

from NN import Softmax, NN


numberOfPoints = 100 #number of data points per class
numberOfClasses = 4 #number of classes in dataset

#data generation: creates spiral dataset with 4 classes and 100 samples each
def generateData(numberOfPoints, numberOfClasses):
    data = np.empty([3, numberOfClasses * numberOfPoints])
    for i in range(0, numberOfClasses):
        data[0, i*numberOfPoints:(i+1)*numberOfPoints] = np.float(i) #np.matrix(np.ones(numberOfPoints));
        radius = np.linspace(0.05, 1, numberOfPoints)
        theta  = np.linspace(i*2*math.pi/numberOfClasses, i*2*math.pi/numberOfClasses + 3*math.pi/2, numberOfPoints) + \
                 np.random.normal(0, .1, numberOfPoints)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        datum = np.matrix(np.transpose(np.column_stack((x, y))))
        data[1:,i*numberOfPoints:i*numberOfPoints + numberOfPoints] = datum
    return data

def display(data):
    #displaying classification data
    fig = plt.figure()
    # for i in range(0, numberOfClasses):
    #     plt.scatter(data[1, i*numberOfPoints:(i+1)*numberOfPoints], data[2,i*numberOfPoints:(i+1)*numberOfPoints])
    plt.scatter(data[1, :], data[2, :], c=data[0, :], s=40, cmap=plt.cm.Spectral)
    plt.show()

def train_softmax(data):
    print(data.T, np.shape(data))
    print(data.T[:, 1:3].shape)
    print(data.T[:, 0].shape)
    softmax = Softmax(data.T[: ,1:3], data.T[:, 0])
    softmax.train()

def train_nn(data):
    print(data.T, np.shape(data))
    print(data.T[:, 1:3].shape)
    print(data.T[range(400), 0].shape)

    nn = NN(data.T[: ,1:3], data.T[:, 0])

    nn.train()


if __name__ == '__main__':
    data = generateData(numberOfPoints, numberOfClasses)
    # train_softmax(data)
    train_nn(data)

    display(data)
