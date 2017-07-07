import math
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf

"""
Implementing a simple one hidden layer on the spiral dataset
"""
class NN:
    def __init__(self, x, y):
        h = 100
        self._w = 0.01 * np.random.randn(x.shape[1], h) # compute hidden layer
        self._b = np.zeros((1,h))
        self._w2 = 0.01 * np.random.randn(h, 4) # computes output
        self._b2 = np.zeros((1, 4))
        self.x = x
        self.y = y
    def eval(self):
        hidden = np.maximum(0, np.dot(self.x, self._w) + self._b) # ReLU activation
        output = np.dot(hidden, self._w2) + self._b2
        return hidden, output
    def loss(self):
        """softmax loss"""
        m = self.x.shape[0]
        exp_scores = np.exp(self.eval()[1])
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(self.x.shape[0]) , self.y.astype(int)])
        data_loss = np.sum(corect_logprobs)/self.x.shape[0]
        reg_loss = 0.5 * 1e-3 * np.sum(self._w * self._w)
        loss = data_loss + reg_loss
        return loss, probs
    def gradients(self):
        hidden, output = self.eval()
        loss, probs = self.loss()
        dscores = probs
        dscores[range(self.x.shape[0]),self.y.astype(int)] -= 1
        dscores /= self.x.shape[0]

        dW2 = np.dot(hidden.T, dscores) #
        db2 = np.sum(dscores, axis=0, keepdims=True)

        dhidden = np.dot(dscores, self._w2.T)
        dhidden[hidden <= 0 ] = 0 # gradient of RELU

        dW = np.dot(self.x.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        dW2 = -1e-3 * self._w2
        dW = -1e-3 * self._w

        self._w += -1e-0 * dW
        self._b += -1e-0 * db
        self._w2 += -1e-0 * dW2
        self._b2 += -1e-0 * db2


    def train(self):
        for i in range(10000):
            loss, _ = self.loss()
            print('iteration #%d: loss %f ' % (i, loss))
            self.gradients()
        _, scores = self.eval()
        predicted_class = np.argmax(scores, axis=1)
        print ('training accuracy: %.2f' % (np.mean(predicted_class == self.y)))

"""
Softmax classifier
"""
class Softmax:
    def __init__(self, x, y):
        self._w = 0.01 * np.random.randn(x.shape[1], 4) # (2 x 4)
        self._b = np.zeros((1, 4))
        self.x = x # (300 x 2)
        self.y = y #labels (300 x 1)
    def eval(self):
        return np.dot(self.x, self._w) + self._b #( 300 x 4)

    def loss(self):
        """softmax loss"""
        m = self.x.shape[0]
        exp_scores = np.exp(self.eval())
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(self.x.shape[0]) , self.y.astype(int)])
        data_loss = np.sum(corect_logprobs)/self.x.shape[0]
        reg_loss = 0.5 * 1e-3 * np.sum(self._w * self._w)
        loss = data_loss + reg_loss
        return loss, probs

    def gradients(self):
        """performs single parameter update"""
        loss, probs = self.loss()
        dscores = probs
        dscores[range(self.x.shape[0]),self.y.astype(int)] -= 1
        dscores /= self.x.shape[0]
        dW = np.dot(self.x.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        dW += 1e-3*self._w # don't forget the regularization gradient
        self._w += -1e-0 * dW
        self._b += -1e-0 * db

    def train(self):
        for i in range(200):
            loss, probs = self.loss()
            print('iteration #%d: loss %f ' % (i, loss))
            self.gradients()
        scores = self.eval()
        predicted_class = np.argmax(scores, axis=1)
        print ('training accuracy: %.2f' % (np.mean(predicted_class == self.y)))







if __name__ == '__main__':
    pass
    # hello = tf.constant('Hello, TensorFlow!')
    #
    # # Start tf session
    # sess = tf.Session()
    #
    # # Run the op
    # print(sess.run(hello))
