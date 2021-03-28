#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, args.data_size)
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)
    bias = np.mean(train_target)

    # there are number of betas equal to data_size

    # TODO: Perform `args.iterations` of SGD-like updates, but in dual formulation
    # using `betas` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is batched MSE with L2 regularization:
    #   L = sum_{i \in B} 1/|B| * [1/2 * (phi(x_i)^T w + bias - target_i)^2] + 1/2 * args.l2 * w^2
    #
    # For `bias`, use explicitly the average of the training targets, and do
    # not update it futher during training.
    #
    # Instead of using feature map `phi` directly, we use a given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each iteration, compute RMSE both on training and testing data.

    def poly_kernel(x, y, gamma, degree):
        # x and y are 1D vectors
        # nonhomogeneous polynomial kernel
        # generate all features until degree
        return ( gamma * np.dot(x, y) + 1 ) ** degree

    def rbf_kernel(x, y, gamma, _):
        # utilize infinite degrees for computation
        # it's similar to knn
        # but this time we are considering whole data as neighbor
        # gamma is controlling how much importance we will give to local points

        exponent = - gamma * (np.linalg.norm(x - y) ** 2)
        return np.exp(exponent)

    def predict(betas, ins, kernel_f, data, bias):
        # predict according to kernel function
        return np.sum([ betas[i] * kernel_f(ins, data[i], args.kernel_gamma, args.kernel_degree) for i in range(len(data))]) + bias


    # pre-calculate kernel
    kernel_func = rbf_kernel if args.kernel == "rbf" else poly_kernel

    TRAIN_KERNEL = np.zeros((args.data_size, args.data_size))

    for i in range(args.data_size):
        for j in range(args.data_size):
            TRAIN_KERNEL[i, j] = kernel_func(train_data[i], train_data[j],
                                       args.kernel_gamma, args.kernel_degree)

    train_rmses, test_rmses = [], []

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`, performing
        # batched updates to the `betas`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.

        # if there is l2 reg update betas according to it after each iter
        betas *= (1 - args.learning_rate * args.l2)

        for i in range(0, args.data_size, args.batch_size):

            until = i + args.batch_size
            batch_indexes = permutation[i:until]

            for exp_i in batch_indexes:
                # for each example instance update the corresponding beta

                update = args.learning_rate * ( train_target[exp_i] - (TRAIN_KERNEL[exp_i] @ betas) - bias)
                betas[exp_i] = betas[exp_i] +  update


        # TODO: Append RMSE on training and testing data to `train_rmses` and
        # `test_rmses` after the iteration.

        # after each iteration calc rmses


        train_predictions = [predict(betas, ins, kernel_func, train_data, bias) for ins in (train_data)]
        test_predictions = [predict(betas, ins, kernel_func, train_data, bias) for ins in (test_data)]

        train_rmses.append( np.sqrt(sklearn.metrics.mean_squared_error(train_target, train_predictions)) )
        test_rmses.append( np.sqrt(sklearn.metrics.mean_squared_error(test_target, test_predictions)) )

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)#!/usr/bin/env python3
