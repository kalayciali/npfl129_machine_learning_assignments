#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.



def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # in this assignment we are using bias explicitly

    train_num_of_instances, num_of_features = train_data.shape

    # Generate initial model weights
    weights = [generator.uniform(size=[num_of_features, args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def relu(x):
        return np.maximum(0, x)


    def softmax(h):
        exp_h = np.exp(h - np.max(h))
        return exp_h / exp_h.sum(axis=0)

    def one_hot_encoding(x, classes):
        x = int(x)
        one_hot = np.zeros(classes)
        one_hot[x] = 1
        return one_hot

        
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

    def forward(inpt, weights, biases):
        # it takes one instance for each time
        # relu non-linear activation
        hidden_layer = relu(inpt @ weights[0] + biases[0])
        out_layer = softmax(hidden_layer @ weights[1] + biases[1])
        return hidden_layer, out_layer

    for iteration in range(args.iterations):

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]
        permutation = generator.permutation(train_num_of_instances)
        grad = [np.zeros_like(weights[0]), np.zeros_like(weights[1])]

        for i, perm in enumerate(permutation):
            inpt, target = train_data[perm], train_target[perm]
            one_hot_target= one_hot_encoding(target, args.classes)

            # instance by instance calculate gradient
            hidden_vals, output_vals = forward(inpt, weights, biases)
            delta = one_hot_target - output_vals
            # gradient1 = (target - output) @ instance
            g = hidden_vals.reshape(-1, 1) @ delta.reshape(1, -1)
            grad[1] += g

            delta = delta @ weights[1].T
            g = (inpt.reshape(-1, 1) @ delta.reshape(1, -1)) * (hidden_vals > 0)
            grad[0] += g

            if (i + 1) % args.batch_size == 0:
                # batch update on weights
                weights[0] += args.learning_rate * grad[0] / args.batch_size
                weights[1] += args.learning_rate * grad[1] / args.batch_size
                grad = [np.zeros_like(weights[0]), np.zeros_like(weights[1])]


        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        predictions = np.asarray([forward(x, weights, biases)[1] for x in test_data])
        test_accuracy = np.mean(predictions.argmax(axis=1) == test_target)
        predictions = np.asarray([forward(x, weights, biases)[1] for x in train_data])
        train_accuracy = np.mean(predictions.argmax(axis=1) == train_target)

        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")

