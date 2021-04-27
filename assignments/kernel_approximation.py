#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import sklearn.datasets
import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.svm

from math import pi

# You could look at this link
# I had used nice explanations of it
# https://maelfabien.github.io/machinelearning/largescale/

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=0.022, type=float, help="RBF gamma")
parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
# see we have large data 
parser.add_argument("--data_size", default=10000, type=int, help="data size")
parser.add_argument("--nystroem", default=0, type=int, help="Use Nystroem approximation")
parser.add_argument("--original", default=False, action="store_true", help="Use original data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--rff", default=0, type=int, help="Use RFFs")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--svm", default=False, action="store_true", help="Use SVM instead of LR")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class RFFsTransformer(sklearn.base.TransformerMixin):
    # Totally random fourier approximation
    # It uses fourier frequency spectrum
    # bias sampled from uniform(0, 2pi)
    # weights sampled from normal(0, sqrt(2 * gamma))

    def __init__(self, n_components, gamma, seed):
        # n_components is number of feats we will approx
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # TODO: Generate suitable `w` and `b`.
        # To obtain deterministic results, generate
        # - `w` first, using a single `generator.normal` call with
        #   output shape `(input_features, self._n_components)`
        # - `b` second, using a single `generator.uniform` call
        #   with output shape `(self._n_components,)`

        n_examples, n_feats = X.shape

        # mean, variance, output_shape
        self._W = generator.normal(0, np.sqrt(2*self._gamma), (n_feats, self._n_components))
        self._b = generator.uniform(0, 2*pi, (1, self._n_components))
        return self

    def transform(self, X):
        # TODO: Transform the given `X` using precomputed `w` and `b`.

        n_feats = X.shape[1]

        X_transformed = np.sqrt(2/n_feats) * np.cos(np.dot(X, self._W) + self._b)
        return X_transformed

class NystroemTransformer(sklearn.base.TransformerMixin):
    # Nystroem approx to Kernel(Gram) Matrix
    # rank-k approx is bigO(n^2) to bigO(n^3)
    # Here we are approximating to rank-k approx of G in linear time
    # With sampling c instances from data

    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def _rbf_kernel(self, X, Z):
        # TODO: Compute the RBF kernel with `self._gamma` for
        # given two sets of examples.
        #
        # A reasonably efficient implementation should probably compute the
        # kernel line-by-line, computing K(X_i, Z) using a single `np.linalg.norm`
        # call, and then concatenate the results using `np.stack`.
        n_samples_X = X.shape[0]
        n_samples_Z = Z.shape[0]
        kernel = np.zeros((n_samples_X, n_samples_Z))
        for i in range(n_samples_X):
            for j in range(n_samples_Z):
                # by default l2 norm
                exponent = - self._gamma * (np.linalg.norm(X[i] - Z[j]) ** 2)
                kernel[i, j] = np.exp(exponent)

        return kernel

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # TODO: Choose a random subset of examples, utilizing indices
        #   indices = generator.choice(X.shape[0], size=self._n_components, replace=False)
        #
        # Then, compute K as the RBF kernel of the chosen examples and
        # V as K^{-1/2} -- use `np.linalg.svd(K, hermitian=True)` to compute
        # the SVD (equal to eigenvalue decomposition for real symmetric matrices).
        # Add 1e-12 to the diagonal of the diagonal matrix returned by SVD
        # before computing the inverse of the square root.

        n_samples = X.shape[0]
        indices = generator.choice(n_samples, size=self._n_components, replace=False)

        X_selected = X[indices, :]

        W = self._rbf_kernel(X_selected, X_selected)


        # we are certain that W will be symmetric matrix
        U, S, Vt = np.linalg.svd(W, hermitian=True, full_matrices=False)

        # if you want to make process faster
        # select k important ones from svd of sampled kernel matrix

        # diagonal could have very near 0 vals 
        # which makes division blow up
        # we need to add some small number just to diagonal
        evals = np.diag(S) + 10 ** -12

        self._M = np.dot(U, 1 / np.sqrt(evals))
        self._train_X = X_selected

        return self

    def transform(self, X):
        # TODO: Compute the RBF kernel of `X` and the chosen training examples
        # and then process it using the precomputed `V`.
        kernel = self._rbf_kernel(X, self._train_X)
        X_transformed = np.dot(kernel, self._M)
        return X_transformed

def main(args):
    # make data for yourself
    X, y = sklearn.datasets.make_classification(n_samples=args.data_size)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(X, y, test_size=args.test_size, random_state=args.seed)


    features = []
    if args.original:
        # like identity transformer
        # when you don't feed any function
        # it doesn't do anything to features
        features.append(("original", sklearn.preprocessing.FunctionTransformer()))
    if args.rff:
        features.append(("rff", RFFsTransformer(args.rff, args.gamma, args.seed)))
    if args.nystroem:
        features.append(("nystroem", NystroemTransformer(args.nystroem, args.gamma, args.seed)))

    if args.svm:
        classifier = sklearn.svm.SVC()
    else:
        classifier = sklearn.linear_model.LogisticRegression(solver="saga", penalty="none", max_iter=args.max_iter, random_state=args.seed)

    pipeline = sklearn.pipeline.Pipeline([
        ("scaling", sklearn.preprocessing.StandardScaler()),
        ("features", sklearn.pipeline.FeatureUnion(features)),
        ("classifier", classifier),
    ])

    pipeline.fit(train_data, train_target)

    test_accuracy = sklearn.metrics.accuracy_score(test_target, pipeline.predict(test_data))
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))

