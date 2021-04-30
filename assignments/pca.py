#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline

class MNIST:
    """MNIST Dataset.
    The train set contains 42000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self, fname="mnist_data.csv"):
        if not (os.path.exists(fname)):
            print("https://www.kaggle.com/c/digit-recognizer/data?select=train.csv")
            raise FileNotFoundError("You can get MNIST data from Kaggle")

        # Load the dataset, i.e., `data` and optionally `target`.
        data = np.loadtxt(fname, skiprows=1, delimiter=",", dtype=np.uint8)
        self.target = data[:, 0]
        # delete target column
        self.data = np.delete(data, obj=0, axis=1)


class PCATransformer(sklearn.base.TransformerMixin):

    def __init__(self, n_components, seed):
        self._n_components = n_components
        self._seed = seed
        self._max_iter = 100000

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        n_instances, n_feats = X.shape
        mean = np.mean(X, axis=0)

        # TODO: Compute the `args._n_components` principal components
        # and store them as columns of `self._V` matrix.
        if self._n_components <= 10:
            # TODO: Use the power iteration algorithm for <= 10 dimensions.
            #
            # To compute every eigenvector, apply 10 iterations, and set
            # the initial value of every eigenvector to
            #   generator.uniform(-1, 1, size=X.shape[1])
            # Compute the vector norms using `np.linalg.norm`.

            # evec matrix
            # cov matrix S
            S = 1 / n_instances * (X - mean).T @ (X-mean)
            V = np.zeros((n_feats, self._n_components))
            zeros = np.zeros(n_feats)

            # power iteration algo
            for i in range(self._n_components):
                evec = generator.uniform(-1, 1, size=n_feats)
                for _ in range(self._max_iter):
                    multip = S @ evec
                    eval = np.linalg.norm(multip)
                    new_evec = multip / eval
                    # 5 decimal place round
                    # if equal to 0 => converged
                    if np.all(np.around(new_evec - evec, 5) == zeros):
                        print("converged for principal ", i + 1)
                        # it converged
                        evec = new_evec
                        break
                    # didn't converged
                    evec = new_evec

                # whether converged or not, found smt
                V[:, i] = evec
                # remove found eval matrix from cov matrix
                S -= eval * (evec @ evec.T)

            self._V = V

        else:
            # TODO: Use the SVD decomposition computed with `np.linalg.svd`
            # to find the principal components.

            U, D, Vt = np.linalg.svd(X - mean, full_matrices=False)
            self._V = Vt.T[:, self._n_components]

        # We round the principal components to avoid rounding errors during
        # ReCodEx evaluation.
        self._V = np.around(self._V, decimals=4)

        return self

    def transform(self, X):
        # TODO: Transform the given `X` using the precomputed `self._V`.
        return X @ self._V

def main(args):
    # Use the MNIST dataset.
    dataset = MNIST()

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    pca = [("PCA", PCATransformer(args.pca, args.seed))] if args.pca else []

    pipeline = sklearn.pipeline.Pipeline(
        [("scaling", sklearn.preprocessing.MinMaxScaler())] +
        pca +
        [("classifier", sklearn.linear_model.LogisticRegression(solver="saga", max_iter=args.max_iter, random_state=args.seed))]
    )
    pipeline.fit(train_data, train_target)

    test_accuracy = pipeline.score(test_data, test_target)
    return test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
    parser.add_argument("--pca", default=None, type=int, help="PCA dimensionality")

    parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
    # If you add more arguments, ReCodEx will keep them with your default values.

    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
