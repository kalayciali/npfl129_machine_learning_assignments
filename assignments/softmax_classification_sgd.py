#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

def get_batch_and_target(batch_indexes, data, target):
    num_of_features = data.shape[1]
    num_of_batch_instances = batch_indexes.size

    batch = np.zeros((num_of_batch_instances, num_of_features))
    batch_target = np.zeros(num_of_batch_instances)

    for i, ind in enumerate(batch_indexes):
        batch[i] = data[ind]
        batch_target[i] = target[ind]

    return batch, batch_target

def one_hot_encoding(i, classes):
    i = int(i)
    one_hot = np.zeros(classes)
    one_hot[i] = 1
    return one_hot

def softmax(z):
    # z shape is (num_of_batch_instances, num_of_classes)
    # result of lin_regression
    # const subtracted from all elem
    # it will not affect soln
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def acc_calc(prediction, target):
    # return the index of max probability
    if target == prediction.argmax():
        return 1
    else:
        return 0

def loss_calc(prediction):
    return prediction @ -np.log(prediction)

def loss_and_acc(data, targets, weights):
    lin_reg = data @ weights
    predictions = softmax(lin_reg)
    loss = [loss_calc(prediction) for prediction in predictions]
    acc = [acc_calc(prediction, target) for prediction, target in zip(predictions, targets)]
    return np.mean(loss), np.mean(acc)

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    # This time we use more than 2 classes
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    # This is the other way of doing hstack
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    train_num_of_instances, num_of_features = train_data.shape
    weights = generator.uniform(size=(num_of_features, args.classes), low=-0.1, high=0.1)

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_num_of_instances)

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        for i in range(0, train_num_of_instances, args.batch_size):
            until = i + args.batch_size

            if until >= train_num_of_instances:
                batch_indexes = permutation[i:]
            else:
                batch_indexes = permutation[i:until]

            batch , batch_target = get_batch_and_target(batch_indexes,
                                                        train_data,
                                                        train_target)
            hot_encoded_batch_target = np.array([one_hot_encoding(label, args.classes) for label in batch_target])
            lin_reg = np.dot(batch, weights) 
            predictions = softmax(lin_reg)
            # predictions shape is (num_of_batch_instances, num_of_classes)
            gradient = np.dot(batch.T,  predictions - hot_encoded_batch_target) / batch.shape[0]
            weights = weights - args.learning_rate * gradient

        # TODO: After the SGD iteration, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_loss, train_accuracy = loss_and_acc(train_data, train_target, weights)
        test_loss, test_accuracy = loss_and_acc(test_data, test_target, weights)

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
    # If you add more arguments, ReCodEx will keep them with your default values.

    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")

