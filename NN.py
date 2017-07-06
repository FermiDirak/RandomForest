import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

"""
Implementing a simple one hidden layer on the spiral dataset
"""

"""
Softmax classifier
"""
class Softmax:
    def __init__(self, x, y):
        self._w = 0.01 * np.random.randn(x.shape()[1], 4) # (300x3)
        self._b = np.zeros((1, K))
        self.x = x # (300 x 2)
        self.y = y #labels (300 x 1)
    def eval(self):
        return np.dot(self._w, self.x) + self._b
    def train(self):
        pass
    def gradientdescent(self):
        pass



if __name__ == '__main__':
    hello = tf.constant('Hello, TensorFlow!')

    # Start tf session
    sess = tf.Session()

    # Run the op
    print(sess.run(hello))
