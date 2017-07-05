import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

"""
Implementing a simple one hidden layer on the spiral dataset
"""


if __name__ == '__main__':
    hello = tf.constant('Hello, TensorFlow!')

    # Start tf session
    sess = tf.Session()

    # Run the op
    print(sess.run(hello))
