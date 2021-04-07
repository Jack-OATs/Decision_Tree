import math
from DecisionNode import DecisionNode
import random
import numpy as np


def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as a list of values being 0 or 1)."""
    # TODO: calculate and return entropy
    pass


def function_B(q):
    """Compute the value of B(q) as per the book's formula and class instruction"""
    # TODO: calculate and return B(q)
    pass


def information_gain(previous_classes, current_classes):
    """Compute the information gain between the previous and current classes (each
    a list of 0 and 1 values)."""
    # TODO: calculate and return information gain
    pass


class OutcomeMetrics():
    def __init__(self, classifier_labels, actual_labels):
        self.classifier_labels = classifier_labels
        self.actual_labels = actual_labels
        self.confusion_matrix = self.__build_confusion_matrix()

    def __build_confusion_matrix(self):
        # format should be [[true_positive, false_negative], [false_positive, true_negative]]
        # TODO: build the confusion matrix as formatted above
        return None

    def get_confustion_matrix(self):
        return self.confusion_matrix

    def precision(self):
        # precision is measured as: true_positive/ (true_positive + false_positive)
        # TODO: calculate and return precision
        pass

    def recall(self):
        #recall is measured as: true_positive/ (true_positive + false_negative)
        # TODO: calculate and return recall
        pass

    def accuracy(self):
        #accuracy is measured as:  correct_classifications / total_number_examples
        # TODO: calculate and return accuracy
        pass



class DecisionTree():
    """Class for automatic tree-building
    and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with an empty root
        and the specified depth limit."""
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """
        Build the tree from root using __build_tree__().
        :param features: A numpy 2D list that contains the features or attributes of the data.
        :param classes: A numpy list that contains the classification or outcome of the data that
                        that is parallel to the features list.
        :return: There is nothing returned but you are to set the root of the tree so decsions can
                 be made later.
        """
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):  
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""
        # TODO: create simple tree by returning the root node
        f = lambda feature: feature[0] > feature[1]
        return DecisionNode(None, None, f)

    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        class_labels = []
        class_labels = [self.root.decide(feature) for feature in features]
        return class_labels
