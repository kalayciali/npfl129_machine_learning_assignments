#!/usr/bin/env python3

# Implementation of simple linear regression 
# by using formula of it

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"

    number_of_instances, features = dataset.data.shape
    bias = np.ones((number_of_instances, 1))
    data = np.hstack((dataset.data, bias))

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, dataset.target, 
                                                                                test_size=args.test_size,
                                                                                random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    inv_of_matrix_mult_with_transpose = np.linalg.inv(np.transpose(X_train) @ X_train)
    weight_vec = ( inv_of_matrix_mult_with_transpose @ np.transpose(X_train) ) @ y_train

    # TODO: Predict target values on the test set

    calc_target_vals = X_test @ weight_vec

    # TODO: Compute root mean square error on the test set predictions
    rmse = sklearn.metrics.mean_squared_error(y_test, calc_target_vals, squared=False)
    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
