from supervisedlearner import SupervisedLearner
import numpy as np

class KNNClassifier(SupervisedLearner):
    def __init__(self, feature_funcs, k):
        super(KNNClassifier, self).__init__(feature_funcs)
        self.k = k
        self.feature_list = []
        self.labels_list = []

    def train(self, anchor_points, anchor_labels):
        """
        :param anchor_points: a 2D numpy array, in which each row is
						      a datapoint, without its label, to be used
						      for one of the anchor points

		:param anchor_labels: a list in which the i'th element is the correct label
		                      of the i'th datapoint in anchor_points

		Does not return anything; simply stores anchor_labels and the
		_features_ of anchor_points.
		"""
        self.feature_list = np.array([self.compute_features(row) for row in anchor_points])
        self.labels_list = np.zeros(len(anchor_labels))
        for row in range(len(anchor_points)):
            self.feature_list[row] = self.compute_features(anchor_points[row])
            self.labels_list[row] = anchor_labels[row]


    def predict(self, x):
        """
        Given a single data point, x, represented as a 1D numpy array,
		predicts the class of x by taking a plurality vote among its k
		nearest neighbors in feature space. Resolves ties arbitrarily.

		The K nearest neighbors are determined based on Euclidean distance
		in _feature_ space (so be sure to compute the features of x).

		Returns the label of the class to which x is predicted to belong.
		"""
        # A list containing the Euclidean distance of x from another point y,
        # each element of which is in the form (distance, y index)

        feature_vector = self.compute_features(x)

        #calculate Euclidean distances
        distance_list = []
        for i in range(len(self.feature_list)):
            distance_list.append((np.square(np.sum((self.feature_list[i] - feature_vector) ** 2)), i))
        distance_list.sort()
        index_list = distance_list[0:self.k]
        labels = []
        for j in range(len(index_list)):
            labels.append(self.labels_list[index_list[j][1]]) # get labels of the shortest distance elements
        return max(set(labels), key=labels.count)
        

    def evaluate(self, datapoints, labels):
        """
        :param datapoints: a 2D numpy array, in which each row is a datapoint.
		:param labels: a 1D numpy array, in which the i'th element is the
		               correct label of the i'th datapoint.

		Returns the fraction (between 0 and 1) of the given datapoints to which
		predict(.) assigns the correct label
		"""
        # Count the number of correct predictions and find the model accuracy
        count = 0
        for row in range(len(datapoints)):
            if (self.predict(datapoints[row]) == labels[row]):
                count+=1
        return count / len(labels)
