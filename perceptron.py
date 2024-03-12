from supervisedlearner import SupervisedLearner
import numpy as np

class Perceptron(SupervisedLearner):
    def __init__(self, feature_funcs, lr, is_c):
        """

        :param lr: the rate at which the weights are modified at each iteration.
        :param is_c: True if the perceptron is for classification problems,
                     False if the perceptron is for regression problems.

        """

        super().__init__(feature_funcs)
        self.weights = None
        self.learning_rate = lr
        self._trained = False
        self.is_classifier = is_c

    def step_function(self, inp):
        """

        :param inp: a real number
        :return: the predicted label produced by the given input

        Assigns a label of 1.0 to the datapoint if <w,x> is a positive quantity
        otherwise assigns label 0.0. Should only be called when self.is_classifier
        is True.
        """
        return 1.0 if inp > 0 else 0.0

    def train(self, X, Y):
        """

        :param X: a 2D numpy array where each row represents a datapoint
        :param Y: a 1D numpy array where i'th element is the label of the corresponding datapoint in X
        :return:

        Does not return anything; only learns and stores as instance variable self.weights a 1D numpy
        array whose i'th element is the weight on the i'th feature.
        """
        self.weights = np.zeros(X.shape[1] + 1)
        X = np.column_stack((X, np.ones(X.shape[0])))
        for i in range(1000):
            for row in range(len(X)):
                label = self.predict(X[row])
                if (self.is_classifier):
                    if (label != Y[row]):
                        self.weights += self.learning_rate * (Y[row] - label) * X[row]
                else:
                    self.weights += self.learning_rate * (Y[row] - label) * X[row]

    def predict(self, x):
        """
        :param x: a 1D numpy array representing a single datapoints
        :return:

        Given a data point x, produces the learner's estimate
        of f(x). Use self.weights and make sure to use self.step_function
        if self.is_classifier is True
        """
        if (len(x) < len(self.weights)):
            x = np.append(x, 1)
        if (self.is_classifier):
            return self.step_function(np.sum(x * self.weights))
        else:
            return np.sum(x * self.weights)

    def evaluate(self, datapoints, labels):
        """

        :param datapoints: a 2D numpy array where each row represents a datapoint
        :param labels: a 1D numpy array where i'th element is the label of the corresponding datapoint in datapoints
        :return:

        If self.is_classifier is True, returns the fraction (between 0 and 1)
        of the given datapoints to which the method predict(.) assigns the correct label
        If self.is_classifier is False, returns the Mean Squared Error (MSE)
        between the labels and the predictions of their respective inputs (You
        do not have to calculate the R2 Score)
        """
        count = 0
        sum = 0
        datapoints = np.column_stack((datapoints, np.ones(datapoints.shape[0])))
        if (self.is_classifier):
            for row in range(len(datapoints)):
                if (self.predict(datapoints[row]) == labels[row]):
                    count += 1
            return count / len(labels)
        else:
            for row in range(len(datapoints)):
                sum += (labels[row] - self.predict(datapoints[row])) ** 2
            return sum / len(labels)
