import argparse
import sys
import math

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def add_ones_to_end(data):
    # data needs to be nparray
    number_of_instances, features = data.shape
    bias = np.ones((number_of_instances, 1))
    data = np.hstack((data, bias))
    return data


def sigmoid(y):
    # return vals between 0 and 1
    # aka probability
    # y is lin_reg result
    return 1/(1 + np.exp(-y))

@np.vectorize
def loss_calc(prediction, target):
    if target == 1:
        return -np.log(prediction)
    else:
        return -np.log(1-prediction)

@np.vectorize
def acc_calc(prediction, target):
    if round(prediction) == target:
        return 1
    else:
        return 0

def mle_loss_and_acc(data, target, weights):
    lin_reg = data @ weights
    loss = loss_calc(sigmoid(lin_reg), target)
    acc = acc_calc(sigmoid(lin_reg), target)
    return np.mean(loss), np.mean(acc)


def get_batch_and_target(batch_indexes, train_data, train_target):

    num_of_features = train_data.shape[1]
    num_of_batch_instances = batch_indexes.size

    batch = np.zeros((num_of_batch_instances, num_of_features ))
    target = np.zeros(num_of_batch_instances)

    at = 0
    for ind in batch_indexes:
        batch[at] = train_data[ind]
        target[at] = train_target[ind]
        at += 1
    return batch, target

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    # target is 0 and 1

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = add_ones_to_end(data)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target= sklearn.model_selection.train_test_split(data, target,
                                                                              test_size=args.test_size,
                                                                              random_state=args.seed)

    train_num_of_instances, train_num_of_features = train_data.shape
    test_num_of_instances, test_num_of_features = test_data.shape

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_num_of_features)

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_num_of_instances)

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        for i in range(0, train_num_of_instances, args.batch_size):
            until = i + args.batch_size

            if until >= train_num_of_instances:
                batch_indexes = permutation[i:]
            else:
                batch_indexes = permutation[i:until]


            batch, batch_target = get_batch_and_target(batch_indexes,
                                                       train_data,
                                                       train_target)
            lin_reg = batch @ weights

            predictions = sigmoid(lin_reg)
            # generalized linear models gradient calculation
            gradient = ( np.dot(batch.T , predictions - batch_target) ) / batch_indexes.size
            weights = weights - args.learning_rate * gradient

        # TODO: After the SGD iteration, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_loss, train_accuracy = mle_loss_and_acc(train_data, train_target, weights)
        test_loss, test_accuracy = mle_loss_and_acc(test_data, test_target, weights)

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                if not iteration: plt.figure(figsize=(6.4*3, 4.8*(args.iterations+2)//3))
                plt.subplot(3, (args.iterations+2)//3, 1 + iteration)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            if args.plot is True: plt.show()
            else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))

