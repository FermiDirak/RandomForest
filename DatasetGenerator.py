import math
import numpy as np

class DatasetGenerator:
    def __init__(self, number_of_points, number_of_classes):
        self.number_of_points = number_of_points
        self.number_of_classes = number_of_classes

    def generateSpiral(self):
        data = np.empty([3, self.number_of_points, self.number_of_classes])

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
