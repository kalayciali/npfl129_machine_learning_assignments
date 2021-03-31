#!/usr/bin/env python3
import argparse

import collections
import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

infintesimal = 0.00001

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
        return counter

    def std_dev(data):
        mean_data = np.mean(data, axis=0)
        num_of_instances = data.shape[0]
        sum_of_differences = np.sum([ (instance - mean_data)**2 for instance in data], axis=0)
        return np.sqrt(sum_of_differences) / float(num_of_instances - 1)

    def normal_dist_probabilities(data, test_data):
        # means of column values
        locs = np.mean(data, axis=0)
        # std_devs of column values
        scales = std_dev(data)

        probs = np.zeros(test_data.shape)
        # iterate over each column
        for i in range(test_data.shape[1]):
            loc = locs[i]
            scale = scales[i] + args.alpha
            calc_prob = lambda val: scipy.stats.norm.pdf(val, loc=loc, scale=scale) * infintesimal
            calc_prob = np.vectorize(calc_prob)
            probs[:, i] = calc_prob(test_data[:, i])
        return probs

    def predict_gaussian(train_data, train_target, test_data):
        data_probs = normal_dist_probabilities(train_data, test_data)
        multiplied_data_probs = np.prod(data_probs, axis=1)
        class_probs = calc_class_prob(train_target)
        probs = np.zeros((data_probs.shape[0], len(class_probs)))

        # assuming here label as int
        for i, prob in enumerate(multiplied_data_probs):
            for label, class_prob in class_probs.items():
                probs[i][label] = class_prob * prob

        pred = np.argmax(probs, axis=1)
        return pred


    predictions = predict_gaussian(train_data, train_target, test_data)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, predictions)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
