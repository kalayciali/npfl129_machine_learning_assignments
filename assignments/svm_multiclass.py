#!/usr/bin/env python3
import argparse
import itertools

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

# Copy paste methods from smo algorithm

def get_data_only_for_binary(perm, data, target):
    # we will get data only for perm classes
    b_data, b_target = [], []
    for d, t in zip(data, target):
        if perm[0] == t:
            b_target.append(1)
            b_data.append(d)
        elif perm[1] == t:
            b_target.append(-1)
            b_data.append(d)

    b_data, b_target = np.array(b_data), np.array(b_target)
    return (b_data, b_target)


def poly_kernel(x, y, gamma, degree):
    return (gamma * np.dot(x,y) + 1) ** degree

def rbf_kernel(x, y, gamma, _):
    exponent = - gamma * (np.linalg.norm(x - y) ** 2)
    return np.exp(exponent)

def predict(a, ins, kernel_f, data, target, bias):
    # predict according to kernel function
    return sum(a[i] * target[i] * kernel_f(ins, data[i], args.kernel_gamma, args.kernel_degree) for i in range(len(a))) + bias


def smo(args, train_data, train_target, test_data):
    # Create initial weights
    # there is one bias and 
    # lagrange multipliers for each data instance
    # if data instance is not support vector
    # its 'a' value is 0

    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    # decide on which kernel func
    kernel_f = poly_kernel if args.kernel == "poly" else rbf_kernel

    # precalculate kernel
    TRAIN_KERNEL = np.zeros((len(train_data), len(train_data)))

    for i in range(len(train_data)):
        for j in range(len(train_data)):
            TRAIN_KERNEL[i, j] = kernel_f(train_data[i], train_data[j], args.kernel_gamma, args.kernel_degree)

    predict_for = lambda ins: predict(a, ins, kernel_f, train_data, train_target, b)


    # we control multiple times 
    # whether found soln is correct 
    passes_without_as_changing = 0
    train_accs, test_accs = [], []


    # for easy calc. introduce error param
    # here we mostly depend on the order of data
    E = [ predict_for(train_data[i]) - train_target[i] for i in range(len(train_data)) ]

    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        # generate random integers array 
        # then enumerate it 
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # we decrease j 1 so we are safe to add 1 to j for each conditions
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.

            # If the conditions do not hold, then
            # - compute the updated unclipped a_j^new.
            #
            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - clip the a_j^new to suitable [L, H].
            #
            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.

            # - increase `as_changed`


            E[i] = predict_for(train_data[i]) - train_target[i]

            # check for KKT conditions
            lower_than_C = a[i] < args.C and train_target[i] * E[i] > - args.tolerance
            higher_than_0 = a[i] > 0 and train_target[i] * E[i] < args.tolerance

            if not lower_than_C or not higher_than_0:
                # required conditions not met 
                E[j] = predict_for(train_data[j]) - train_target[j]

                second_deriv = 2 * TRAIN_KERNEL[i, j] - TRAIN_KERNEL[i, i] - TRAIN_KERNEL[j, j]

                if second_deriv > -args.tolerance:
                    # second deriv is positive
                    # concave up
                    # pass it
                    # we are looking for maximizing lagrange func
                    continue


                # a_i + ( t_i * t_j * a_j ) = some_const
                # here we are arranging lower and higher bounds 
                # lagrange multip needs to be between C and 0 
                # in order to meet with KKT conditions 

                L = max(0, a[j] - a[i]) if train_target[i] != train_target[j] else max(0, a[i] + a[j] - args.C)
                H = min(args.C, args.C + a[j] - a[i]) if train_target[i] != train_target[j] else  min(args.C, a[i] + a[j])
                if (H - L < args.tolerance):
                    continue

                # compute new a_j
                a_j_new = a[j] - train_target[j] * (E[i] - E[j]) / second_deriv
                a_j_new = min(H, a_j_new)
                a_j_new = max(L, a_j_new)

                if (np.abs(a[j] - a_j_new) < args.tolerance):
                    continue

                a_i_new = a[i] - train_target[i] * train_target[j] * (a_j_new - a[j])
                b_j = b - E[j] - (train_target[i] * (a_i_new - a[i]) * TRAIN_KERNEL[i, j]) - (train_target[j] * (a_j_new - a[j]) * TRAIN_KERNEL[j, j])
                b_i = b - E[i] - (train_target[i] * (a_i_new - a[i]) * TRAIN_KERNEL[i, i]) - (train_target[j] * (a_j_new - a[j]) * TRAIN_KERNEL[j, i])

                if 0 < a_i_new < args.C:
                    b = b_i
                elif 0 < a_j_new < args.C:
                    b = b_j
                else:
                    b = (b_i + b_j) * 0.5

                a[i] = a_i_new
                a[j] = a_j_new
                as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.

        train_predictions = [ 1 if predict_for(train_data[i]) > 0 else -1 for i in range(len(train_data)) ]
        train_accs.append(sklearn.metrics.accuracy_score(train_target, train_predictions))

        test_predictions = [ 1 if predict_for(test_data[i]) > 0 else -1 for i in range(len(test_data)) ]


        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 10 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}% ".format(
                len(train_accs), 100 * train_accs[-1]))

    print("Training finished after iteration {}, train acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    # Note that until now the full `a` should have been used for prediction.

    return test_predictions


def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.
    # so we need to write method to change target vals


    votes = np.zeros(shape=(len(test_data), args.classes))
    for perm in itertools.permutations(range(args.classes), 2):
        print(f"Training starting for classes: {perm}")

        b_data, b_target = get_data_only_for_binary(perm, train_data, train_target)
        test_predictions = smo(args, b_data, b_target, test_data)
        # test_predictions are 1 , -1 
        # if target is class perm[0] => pred = 1
        # if target is class perm[1] => pred = -1
        for i in range(len(test_predictions)):
            # voting according to test predictions
            if test_predictions[i] == 1:
                votes[i][perm[0]] += 1
            else:
                votes[i][perm[1]] += 1


    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Note that during prediction, only the support vectors returned by the `smo`
    # should be used, not all training data.
    #
    # Finally, compute the test set prediction accuracy.

    # return index of max vote for each row
    predictions = np.argmax(votes, axis=1)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, predictions)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
