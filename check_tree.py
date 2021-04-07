import numpy as np
import utilities
from decision_trees import DecisionTree, OutcomeMetrics


def run_test_data_on_tree():
    dataset = utilities.load_csv('Student Data.csv')
    print("---------------")
    ten_folds = utilities.generate_k_folds(dataset, 10)
    print("---------------")
    accuracies = []
    precisions = []
    recalls = []
    confusion = []

    for fold in ten_folds:
        train, test = fold
        train_features, train_classes = train
        test_features, test_classes = test
        tree = DecisionTree()
        tree.fit(train_features, train_classes)
        output = tree.classify(test_features)
        outcomes = OutcomeMetrics(output, test_classes)
        accuracies.append(outcomes.accuracy())
        precisions.append(outcomes.precision())
        recalls.append(outcomes.recall())
        confusion.append(outcomes.get_confustion_matrix())

    print("trees average = " + str(np.average(accuracies)))

run_test_data_on_tree()
