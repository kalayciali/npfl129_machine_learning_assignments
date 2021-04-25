#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

def main(args):
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    # we only interested in whether it's odd or even
    # even : 0 , odd : 1
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data,
                                                                                                dataset.target,
                                                                                                test_size=args.test_size,
                                                                                                random_state=args.seed)
    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    minmax = sklearn.preprocessing.MinMaxScaler()
    poly_feat = sklearn.preprocessing.PolynomialFeatures()
    logistic_regress = sklearn.linear_model.LogisticRegression(random_state=args.seed,
                                                               max_iter=10000,
                                                               tol=0.1)
    pipe = sklearn.pipeline.Pipeline([('minmax', minmax),
                                      ('poly', poly_feat),
                                      ('logistic', logistic_regress)])

    # logistic_C is regularization param
    param_grid = {
        'poly__degree': [1, 2],
        'logistic__C': [0.01, 1, 100],
        'logistic__solver': ['lbfgs', 'sag'],
    }
    cv = sklearn.model_selection.StratifiedKFold(5)
    search = sklearn.model_selection.GridSearchCV(pipe, 
                                                  param_grid, 
                                                  n_jobs=-1,
                                                  scoring='accuracy',
                                                  cv=cv)
    search.fit(train_data, train_target)
    test_accuracy = search.score(test_data, test_target)
    print(search.best_params_)
    return test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))

