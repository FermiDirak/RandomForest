import math
import numpy as np
import matplotlib.pyplot as plt

#data generation: creates spiral dataset with 4 classes and 100 samples each

numberOfPoints = 100 #number of data points per class
numberOfClasses = 4 #number of classes in dataset

data = np.empty([2, numberOfClasses * numberOfPoints])

for i in range(0, numberOfClasses):
    radius = np.linspace(0.05, 1, numberOfPoints)
    theta  = np.linspace(i*2*math.pi/numberOfClasses, i*2*math.pi/numberOfClasses + 3*math.pi/2, numberOfPoints) + \
             np.random.normal(0, .1, numberOfPoints)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    datum = np.matrix(np.transpose(np.column_stack((x, y))))
    data[:,i*numberOfPoints:i*numberOfPoints + numberOfPoints] = datum

print(data.size)

fig = plt.figure()
plt.scatter(data[0,0:200], data[1,0:200])
plt.show()
