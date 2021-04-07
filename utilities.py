import random
import numpy as np


def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])
    classes= map(int,  out[:, class_index])
    features = out[:, :class_index]
    return features, classes


def generate_k_folds(dataset, k):
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    examples = list(zip(dataset[0], dataset[1]))
    # print(examples[0])
    random.shuffle(examples)
    # print(examples[0])
    folds = np.array_split(examples, k)
    # up to this point I have dataset split into k groups
    final_folds = []
    for i in range(len(folds)):
        training_examples = []
        training_classes = []
        for j in range(len(folds)):
            if i != j:
                for item in folds[j]:
                    training_examples.append(item[0])
                    training_classes.append(item[1])
        testing_examples = []
        testing_classes = []
        for item in folds[i]:
            testing_examples.append(item[0])
            testing_classes.append(item[1])
        final_folds.append(((training_examples, training_classes), (testing_examples, testing_classes)))
    return final_folds
