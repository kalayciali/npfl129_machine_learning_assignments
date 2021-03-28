#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def poly_kernel(x, y, gamma, degree):
    return (gamma * np.dot(x,y) + 1) ** degree

def rbf_kernel(x, y, gamma, _):
    exponent = - gamma * (np.linalg.norm(x - y) ** 2)
    return np.exp(exponent)

def predict(a, ins, kernel_f, data, target, bias):
    # predict according to kernel function
    return sum(a[i] * target[i] * kernel_f(ins, data[i], args.kernel_gamma, args.kernel_degree) for i in range(len(a))) + bias


# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, train_data, train_target, test_data, test_target):
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
                    # concave
                    continue


                # a_i + ( t_i * t_j * a_j ) = some_const
                # here we are arranging lower and higher bounds 
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
        test_accs.append(sklearn.metrics.accuracy_score(test_target, test_predictions))

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    print("Training finished after iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    # Note that until now the full `a` should have been used for prediction.

    support_vectors, support_vector_weights  = [], []
    for i in range(len(train_data)):
        if a[i] > args.tolerance:
            support_vectors.append(train_data[i])
            support_vector_weights.append(train_target[i] * a[i])
    support_vectors = np.array(support_vectors)
    support_vector_weights = np.array(support_vector_weights)

    return support_vectors, support_vector_weights, a, b, train_accs, test_accs

def main(args):
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, a, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:

        def plotter(predict, support_vectors, data):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        kernel_f = poly_kernel if args.kernel == "poly" else rbf_kernel
        predict_for = lambda ins: predict(a, ins, kernel_f, train_data, train_target, bias)
        plotter(predict_for, support_vectors, data)
        if args.plot is True: plt.show()

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
