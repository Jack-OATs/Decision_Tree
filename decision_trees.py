import math
from decisiontree.DecisionNode import DecisionNode
import random
import numpy as np


def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as a list of values being 0 or 1)."""
    Vk = []
    if class_vector.count(1) == 0 or class_vector.count(0) == 0:
        return 0
    Vk.append(-(class_vector.count(1) / len(class_vector)) * math.log((class_vector.count(1) / len(class_vector)), 2))
    Vk.append(-(class_vector.count(0) / len(class_vector)) * math.log((class_vector.count(0) / len(class_vector)), 2))
    return math.fsum(Vk)


def function_B(q):
    """Compute the value of B(q) as per the book's formula and class instruction"""
    if q == 0 or q == 1:
        return 0
    elif q == 0.5:
        return 1
    else:
        return -((q * math.log(q, 2)) + ((1 - q) * math.log(1 - q, 2)))


def information_gain(previous_classes, current_classes):
    """Compute the information gain between the previous and current classes (each
    a list of 0 and 1 values)."""
    B_pc = function_B(previous_classes.count(1) / (previous_classes.count(1) + previous_classes.count(0)))
    remainder = []
    for val in current_classes:
        if len(val) == 0:
            remainder.append(0)
        else:
            denom1 = (function_B(val.count(1) / (len(val))))
            denom2 = len(previous_classes)
            if denom1 == 0:
                remainder.append(0)
            elif denom2 == 0:
                remainder.append(0)
            else:
                remainder.append((len(val)/denom2) * denom1)
    return B_pc - math.fsum(remainder)


class OutcomeMetrics():
    def __init__(self, classifier_labels, actual_labels):
        self.classifier_labels = classifier_labels
        self.actual_labels = actual_labels
        self.confusion_matrix = self.__build_confusion_matrix()

    def __build_confusion_matrix(self):
        # format should be [[true_positive, false_negative], [false_positive, true_negative]]
        all_labels = zip(self.classifier_labels, self.actual_labels)
        con_matrix = [[0, 0], [0, 0]]
        for cl, al in all_labels:
            if cl == al == 1:
                con_matrix[0][0] += 1
            elif cl == al == 0:
                con_matrix[1][1] += 1
            elif cl == 1 != al:
                con_matrix[1][0] += 1
            elif cl == 0 != al:
                con_matrix[0][1] += 1
        return con_matrix

    def get_confustion_matrix(self):
        return self.confusion_matrix

    def precision(self):
        # precision is measured as: true_positive/ (true_positive + false_positive)
        con_mat = self.get_confustion_matrix()
        tp = con_mat[0][0]
        fp = con_mat[1][0]
        if tp+fp == 0:
            return 0
        return tp / (tp + fp)

    def recall(self):
        # recall is measured as: true_positive/ (true_positive + false_negative)
        con_mat = self.get_confustion_matrix()
        tp = con_mat[0][0]
        fn = con_mat[0][1]
        if tp+fn == 0:
            return 0
        return tp / (tp + fn)

    def accuracy(self):
        # accuracy is measured as:  correct_classifications / total_number_examples
        con_mat = self.get_confustion_matrix()
        tp = con_mat[0][0]
        tn = con_mat[1][1]
        total = math.fsum(con_mat[0]) + math.fsum(con_mat[1])
        if total == 0:
            return 0
        return (tp + tn) / total


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
        best_left_child = []
        best_right_child = []
        cl_count = 0
        v = None
        ind = 0
        best_fun = None
        fn_list = [lambda ft, avg, index: ft[index] > avg[index],
                   lambda ft, med, index: ft[index] >= med,
                   lambda ft, _, index: ft[index] >= 2]
        val_list = [np.average(features, axis=0),
                    np.median(features, axis=0),
                    np.mean(features, axis=0)]
        fc_list = zip(features, classes)
        fv_list = zip(fn_list, val_list)
        max_info = 0

        for fun, val in fv_list:
            # print("val = {}".format(val))
            left_child = []
            right_child = []
            left_child_classes = []
            right_child_classes = []
            for i in range(len(features[0])):
                for rec, cl in fc_list:
                    if fun(rec, val, i):
                        left_child.append(rec)
                        left_child_classes.append(cl)
                        if cl == 0:
                            cl_count -= 1
                        else:
                            cl_count += 1
                    else:
                        right_child.append(rec)
                        right_child_classes.append(cl)
                        if cl == 0:
                            cl_count -= 1
                        else:
                            cl_count += 1
                gain = information_gain(classes, [left_child_classes, right_child_classes])
                if cl_count >= 0:
                    consistent = 1
                else:
                    consistent = 0

                if max_info < gain:
                    max_info = gain
                    best_fun = fun
                    v = val
                    ind = i
                    # best_left_child = left_child
                    # best_right_child = right_child
                if max_info == 0:
                    return DecisionNode(None, None, None, consistent)
                # print("max is {} left_child is {}".format(max_info, left_child))
                # print("max is {} right_child is {}".format(max_info, right_child))
                if not left_child and not right_child:
                    return DecisionNode(None, None, None, consistent)
                elif not left_child:
                    return DecisionNode(None, right_child, None, consistent)
                elif not right_child:
                    return DecisionNode(left_child, None, None, consistent)
                else:
                    best_left_child = self.__build_tree__(left_child, left_child_classes)
                    best_right_child = self.__build_tree__(right_child, right_child_classes)

        return DecisionNode(best_left_child, best_right_child, (best_fun, v, ind))

    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        # print(type(features))
        # print(type(features[0]))
        class_labels = [self.root.decide(feature) for feature in features]
        results = [1 if random.random() <= label else 0 for label in class_labels]
        return results


if __name__ == "__main__":
    om = OutcomeMetrics(None, None)
    dt = DecisionTree()
    dt.__build_tree__()
    # om.accuracy()
