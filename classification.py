'''
SL Assignment - classification.py
CS 1410 Artificial Intelligence, Brown University
Written by whackett.

Usage-

To run classification using your perceptron implementation:
    python classification.py

To run classification using our KNN implementation:
    python classification.py -knn (or python classification.py -k)

'''
from sklearn import preprocessing
from knn import KNNClassifier
from perceptron import Perceptron
from cross_validate import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def classification(method, train_size=0.9):
    """
    Classifies data using the model type specified in method. Default is the
    perceptron.

    Returns the model accuracy on the test data.
    """

    # Load the data. Each row represents a datapoint, consisting of all the
    # feature values and a label, which indicates the class to which the point
    # belongs. Here, each row is a patient. The features are calculated
    # from a digital image of a fine needle aspirate (FNA) of a breast mass, and
    # the label represents the patient's diagnosis, i.e. malignant or benign.
    all_data = pd.read_csv("breast_cancer_diagnostic.csv")

    # Remove the id and Unnamed:32 columns. 
    all_data = all_data.drop(['Unnamed: 32', 'id'], axis = 1)

    # Convert the diagnosis values M and B to numeric values, such that
    # M (malignant) = 1 and B (benign) = 0
    def convert_diagnosis(diagnosis):
        if diagnosis == "B":
            return 0
        else:
            return 1
    all_data["diagnosis"] = all_data["diagnosis"].apply(convert_diagnosis)

    # Store the features of the data
    X = np.array(all_data.iloc[:, 1:])
    # Store the labels of the data
    y = np.array(all_data["diagnosis"])

    train_size = 0.8
    X_train, X_test, y_train, y_test = cross_validate(X, y, train_size)

    print("-" * 30)
    if method == "KNN":
        # Set the number of neighboring points to compare each datapoint to.
        k = 9

        # Normalize the feature data, so that values are between [0,1]. This allows
        # us to use euclidean distance as a meaningful metric across features.
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)

        # For KNN, we want the feature function to return the value of the
        # given feature.
        def feature_func(x):
            return x

        # Initialize the KNN Classifier
        classifier = KNNClassifier([feature_func], k)

        # optimal_k(feature_func, X_train, X_test, y_train, y_test)
    else:
        learning_rate = 0.1
        is_classifier = True
        def feature_func(x):
            return x
        classifier = Perceptron([feature_func], learning_rate, is_classifier)

    # Fit the data on the train set
    print("Training {} Classifier".format(method))
    classifier.train(X_train, y_train)

    # Evaluate the model's accuracy (between 0 and 1) on the test set
    print("Testing {} Classifier".format(method))
    accuracy = classifier.evaluate(X_test, y_test)

    print("{} Model Accuracy: {:.2f}%".format(method, accuracy*100))
    print("-" * 30)

    optimal_k(feature_func, X_train, X_test, y_train, y_test)
    return accuracy


def optimal_k(feature_func, X_train, X_test, y_train, y_test):
    """
    1) Finds the optimal value of k, where k is the number of neighbors being
    looked at during KNN.
    2) Plots the accuracy values returned by performing cross validation on
    the KNN model, with k values in the range [1, 50).
    """

    # Feel free to delete the following commented stencil code.
    # It is only here to help you implement and visualize optimal_k.
    neighbors = []
    accuracies = []
    for k in range(1, 50):
        classifier = KNNClassifier([feature_func], k)
        classifier.train(X_train, y_train)
        neighbors.append(k)
        accuracies.append(classifier.evaluate(X_test, y_test))
    max_index = accuracies.index(max(accuracies))
    print(neighbors[max_index])

    # Visualizing number of neighbors vs accuracy, if neighbors is a list of
    # your values of k and accuracies is a list of the same size,
    # corresponding to the cross validation accuracy returned for a given
    # value of k:
    plt.figure(figsize = (10, 6))
    plt.plot(neighbors, np.multiply(accuracies,100))
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy (% Correct)')
    plt.show()
    quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised Learning - Classification")
    parser.add_argument("-k", "--knn", help="Indicates to use KNN model. Otherwise, uses perceptron.",
        action="store_true")
    args = vars(parser.parse_args())
    if args["knn"]:
        method = "KNN"
    else:
        method = "Perceptron"
    classification(method)
