#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from collections import deque
from collections import Counter

class MNIST:
    """MNIST Dataset.
    The train set contains 42000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self, data="mnist_data.csv"):
        if not (os.path.exists(data)):
            raise FileNotFoundError("You can get MNIST data from Kaggle")

        # Load the dataset, i.e., `data` and optionally `target`.
        data = pd.read_csv(data)
        self.target = data['label']

        data.drop(['label', ], axis=1, inplace=True)
        self.data = data

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=1000, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--train_size", default=1000, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--weights", default="uniform", type=str, help="Weighting to use (uniform/inverse/softmax)")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load MNIST data, scale it to [0, 1] and split it to train and test
    mnist = MNIST()
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, stratify=mnist.target, train_size=args.train_size, test_size=args.test_size, random_state=args.seed)

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    #
    # Find `args.k` nearest neighbors, choosing the ones with smallest train_data
    # indices in case of ties. Use the most frequent class (optionally weighted
    # by a given scheme described below) as prediction, choosing the one with the
    # smallest class index when there are multiple classes with the same frequency.
    #
    # Use L_p norm for a given p (1, 2, 3) to measure distances.
    #
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight
    # - "inverse": `1/distances` is used as weights
    # - "softmax": `softmax(-distances)` is uses as weights
    #
    # If you want to plot misclassified examples, you need to also fill `test_neighbors`
    # with indices of nearest neighbors; but it is not needed for passing in ReCodEx.


    test_predictions = []
    test_neighbors = []
    for test_i, test_instance in enumerate(test_data):
        distances = []
        smallest_k_dist_targets = []
        for train_i, train_instance in enumerate(train_data):
            distance = np.linalg.norm(test_instance - train_instance)
            distances.append((train_i, distance))

        distances.sort(key=lambda x: x[1])

        test_neighbor = []
        for i in range(args.k):
            train_target_i = distances[i][0]
            smallest_k_dist_targets.append(train_target.iloc[train_target_i])
            test_neighbor.append(train_target_i)

        test_neighbors.append(test_neighbor)
        test_predictions.append(max(set(smallest_k_dist_targets), key=smallest_k_dist_targets.count))


    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))

