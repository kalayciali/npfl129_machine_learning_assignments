#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def add_ones_to_end(data):
    # data needs to be ndarray
    number_of_instances, features = data.shape
    bias = np.ones((number_of_instances, 1))
    data = np.hstack((data, bias))
    return data

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = add_ones_to_end(data)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target,
                                                                        test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1])

    train_rmses, test_rmses = [], []

    for epoch in range(args.epochs):

        permutation = generator.permutation(train_data.shape[0])
        # permute the indexes 

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # A gradient for example (x_i, t_i) is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is `weights = weights - args.learning_rate * gradient`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        for i in range(0, train_data.shape[0] , args.batch_size):
            # it handles also if data // batch != 0

            until = i + args.batch_size

            if until >= train_data.shape[0]:
                batch_indexes = permutation[i:]
            else:
                batch_indexes = permutation[i:until]

            num_of_batch_instances = batch_indexes.size

            batch = np.zeros((num_of_batch_instances, train_data.shape[1]))
            target = np.zeros(num_of_batch_instances)

            at = 0
            for ind in batch_indexes:
                batch[at] = train_data[ind]
                target[at] = train_target[ind]
                at += 1


            loss = (batch @ weights - target)

            gradients = np.zeros((num_of_batch_instances, train_data.shape[1]))
            for ind in range(num_of_batch_instances):
                # without regularization
                gradients[ind] = loss[ind] * batch[ind]

            avg_gradient = np.average(gradients, axis=0)
            weights = weights - args.learning_rate * avg_gradient


        # TODO: Append current RMSE on train/test to train_rmses/test_rmses.
        calc_train = train_data @ weights
        calc_target = test_data @ weights
        train_rmse = sklearn.metrics.mean_squared_error(train_target, calc_train, squared=False)
        train_rmses.append(train_rmse)
        test_rmse = sklearn.metrics.mean_squared_error(test_target, calc_target, squared=False)
        test_rmses.append(test_rmse)


    # TODO: Compute into `explicit_rmse` test data RMSE when
    # fitting `sklearn.linear_model.LinearRegression` on train_data.
    model = sklearn.linear_model.LinearRegression()
    model.fit(train_data, train_target)
    calc_target = model.predict(test_data)
    explicit_rmse = sklearn.metrics.mean_squared_error(test_target, calc_target, squared=False)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return test_rmses[-1], explicit_rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))

