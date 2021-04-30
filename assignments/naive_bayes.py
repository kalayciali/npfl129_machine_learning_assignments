#!/usr/bin/env python3
import argparse

import collections
import numpy as np
import math

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

# infinitesimal step to calc prob. from PDF

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)


    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Do not forget that Bernoulli NB works with binary data, so consider
    #   all non-zero features as ones during both estimation and prediction.

    # TODO: Predict the test data classes and compute test accuracy.

    def calc_class_prob(target):
        counter = collections.Counter(target)
        total_sum = sum(counter.values())

        for target in counter:
            counter[target] = counter[target]/ total_sum
        # base class probabilities
        return counter

    def std_dev(data):
        # axis 0 is column direction
        mean_data = np.mean(data, axis=0)
        num_of_instances = data.shape[0]
        sum_of_differences = np.sum(np.array([ (instance - mean_data)**2 for instance in data]), axis=0)
        return np.sqrt(sum_of_differences) / float(num_of_instances - 1)

    def get_feat_mean_var_for_classes(data, target):

        classes = np.unique(target)
        mean_var_of_clms = {}
        for c in classes:
            data_c = data[np.where(target == c)]
            # means of column values
            locs = np.mean(data_c, axis=0)
            # std_devs of column values
            scales = std_dev(data_c)
            mean_var_of_clms[c] = (locs, scales)

        return mean_var_of_clms

    @np.vectorize
    def calc_gaussian(x, mean, var):
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + args.alpha)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + args.alpha)))
        return coeff * exponent

    def calc_data_prob(data, locs, scales):
        # calc_prob with 
        # certain class means and std_vars
        probs = np.zeros(data.shape)
        # iterate over each column
        for i in range(test_data.shape[1]):
            loc = locs[i]
            scale = scales[i] + args.alpha
            probs[:, i] = calc_gaussian(test_data[:, i], loc, scale)
        return probs

    def fit(train_data, train_target):

        mean_var_of_clms = get_feat_mean_var_for_classes(train_data, train_target)
        class_probs = calc_class_prob(train_target)
        return mean_var_of_clms, class_probs

    def predict(test_data, mean_var_of_clms, class_probs):
        # i didn't used class so thats why passing things around

        biggest_probs = None
        predictions = None

        for c in class_probs:
            # assuming target is some integer
            base_prob = class_probs[c]
            mean_var = mean_var_of_clms[c] 
            data_probs = calc_data_prob(test_data, *mean_var)
            probs = np.multiply(base_prob, np.prod(data_probs, axis=1))

            if biggest_probs is None:
                biggest_probs = probs
                predictions = np.full_like(biggest_probs, int(c), dtype=np.int8)
                continue

            inds = np.nonzero(probs > biggest_probs)
            predictions[inds] = int(c)
            biggest_probs[inds] = probs[inds]

        return predictions

    mean_var_of_clms, class_probs = fit(train_data, train_target)
    predictions = predict(test_data, mean_var_of_clms, class_probs)

    test_accuracy = sklearn.metrics.accuracy_score(test_target, predictions)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
