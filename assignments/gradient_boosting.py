#!/usr/bin/env python3
import argparse
import progressbar
from recordclass import recordclass

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

# if leaf, it has value
Node = recordclass('Node', ['left', 'right', 'feat_split', 'optimal_weight'],
                   defaults=[None, None, None, None])

class LogisticLoss():

    # since target is one hot encoded
    # sigmoid function here

    @classmethod
    def sigmoid(cls, y):
        return 1 / (1 + np.exp(-y))

    @classmethod
    def gradient(cls, true_y, pred_y):
        # gradient of loss wrt pred_y
        prob = cls.sigmoid(pred_y)
        return - (true_y - prob)

    @classmethod
    def hessian(cls, true_y, pred_y):
        # second deriv wrt pred_y
        prob = cls.sigmoid(pred_y)
        return prob * (1 - prob) 


class GradientTree():
    # calculation of criterion is handled by 
    # hessian and gradient of loss function

    def __init__(self, max_depth=None, l2=1., min_to_split=2):
        # our only heuristic is max_depth
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.l2 = l2
        self.root = None
        self.feature_names = None
        self.loss = LogisticLoss

    def _calc_gain(self, true_y, pred_y):
        # calculation of information gain 
        # each time splitted data is coming
        sum_of_grad = self.loss.gradient(true_y, pred_y).sum()
        nominator = np.power(sum_of_grad, 2)
        denominator = self.l2 + self.loss.hessian(true_y, pred_y).sum()
        return - 0.5 * (nominator / denominator)

    def _gen_feature_splits(self, data):

        # - When splitting a node, consider the features in sequential order, then
        #   for each feature consider all possible split points ordered in ascending
        #   value, and perform the first encountered split decreasing the criterion
        #   the most. Each split point is an average of two nearest unique feature values
        #   of the instances corresponding to the given node (e.g., for four instances
        #   with values 1, 7, 3, 3 the split points are 2 and 5).

        feature_splits = []
        for clm in data.T:
            sorted_clm = np.unique(clm)
            moving_avg = np.convolve(sorted_clm, np.ones(2), 'valid') / 2.0
            feature_splits.append(moving_avg)
        return feature_splits

    def _leaf_weights(self, true_y, pred_y):
        gradient = np.sum(self.loss.gradient(true_y, pred_y), axis=0)
        hessian = np.sum(self.loss.hessian(true_y, pred_y), axis=0)
        return - gradient / (self.l2 + hessian)

    def _split_data(self, data, true_y, pred_y):

        best_gain = 0
        best_split = None
        best_l = None
        best_r = None

        feat_splits = self._gen_feature_splits(data)

        root_gain = self._calc_gain(true_y, pred_y)

        for feat, splits_to_consider in enumerate(feat_splits):
            for split in splits_to_consider:

                l_ind = (data[:, feat] <= split).nonzero()[0]
                l_gain = self._calc_gain(true_y[l_ind], pred_y[l_ind])

                r_ind = (data[:, feat] > split).nonzero()[0]
                r_gain = self._calc_gain(true_y[r_ind], pred_y[r_ind])

                gain_if_split = l_gain + r_gain - root_gain

                if gain_if_split > best_gain:
                    best_gain = gain_if_split
                    best_split = (feat, split)

                    best_l = (data[l_ind], true_y[l_ind], pred_y[l_ind])
                    best_r = (data[r_ind], true_y[r_ind], pred_y[r_ind])

        return best_l, best_r, best_split


    def _construct_tree(self, data, true_y, pred_y, depth=0):
        # only recursive implementation
        # don't consider max_leaves option
        if data.shape[0] >= self.min_to_split and depth < self.max_depth:

            best_l, best_r, best_split = self._split_data(data, true_y, pred_y)

            if best_l:
                # splitted
                tree = Node(self._construct_tree(*best_l, depth + 1),
                            self._construct_tree(*best_r, depth + 1),
                            best_split, None)
                return tree

            else:
                # conditions are available for splitting
                # but there is no way of splitting 
                # calc leaf weights
                optimal_weight = self._leaf_weights(true_y, pred_y)
                leaf = Node(None, None, None, optimal_weight)
                return leaf
        else:
            # again not splitted 
            # leaf
            optimal_weight = self._leaf_weights(true_y, pred_y)
            leaf = Node(None, None, None, optimal_weight)
            return leaf


    def fit(self, X, true_y, pred_y, feature_names=None):
        # take predictions of previous tree also
        self.feature_names = feature_names
        self.root = self._construct_tree(X, true_y, pred_y)
        return self

    def _predict_one(self, ins):
        current_node = self.root

        while current_node.feat_split:
            feat_i, val = current_node.feat_split
            if ins[feat_i] <= val:
                current_node = current_node.left
            else:
                current_node = current_node.right

        return current_node.optimal_weight

    def predict(self, data):
        predictions = np.apply_along_axis(self._predict_one, 1, data)
        return predictions


class GradientBoostedClassifier():

    # TODO: Create a gradient boosted trees on the classification training data.
    #
    # Notably, train for `args.trees` iteration. During iteration `t`:
    # - the goal is to train `classes` regression trees, each predicting
    #   raw weight for the corresponding class.
    # - compute the current predictions `y_t(x_i)` for every training example `i` as
    #     y_t(x_i)_c = \sum_{i=1}^t args.learning_rate * tree_{iter=i,class=c}.predict(x_i)
    #     (note that y_0 is zero)
    # - loss in iteration `t` is
    #     L = (\sum_i NLL(onehot_target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
    #         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
    # - for every class `c`:
    #   - start by computing `g_i` and `h_i` for every training example `i`;
    #     the `g_i` is the first derivative of NLL(onehot_target_i_c, softmax(y_{t-1}(x_i))_c)
    #     with respect to y_{t-1}(x_i)_c, and the `h_i` is the second derivative of the same.
    #   - then, create a decision tree minimizing the above loss L. According to the slides,
    #     the optimum prediction for a given node T with training examples I_T is
    #       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
    #     and the value of the loss with the above prediction is
    #       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
    #     which you should use as a splitting criterion.
    #
    # During tree construction, we split a node if:
    # - its depth is less than `args.max_depth`
    # - there is more than 1 example corresponding to it (this was covered by
    #     a non-zero criterion value in the previous assignments)

    # TODO: Finally, measure your training and testing accuracies when
    # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (and the smallest class if there is a tie).

    def __init__(self, trees=1, l2=1., learning_rate=0.1,
                 max_depth=None, min_to_split=2):
        # our only heuristic is max_depth
        self.learning_rate = learning_rate
        self.bar = progressbar.ProgressBar()
        self.n_estimators = trees

        self.trees = []
        for _ in range(trees):
            tree = GradientTree(max_depth, l2, min_to_split)
            self.trees.append(tree)

    def _softmax(self, z):
        # z is result of linear regression
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis=0)

    def _one_hot_encoding(self, y, n_cols=None):
        if not n_cols:
            n_cols = np.amax(y) + 1
        one_hot = np.zeros((y.shape[0], n_cols))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


    def fit(self, data, target):
        target = self._one_hot_encoding(target)

        y_pred = np.zeros_like(target)
        for t in self.bar(range(self.n_estimators)):
            tree = self.trees[t]
            tree.fit(data, target, y_pred)
            update = tree.predict(data)

            # did max gain, now we are subtracting
            y_pred += np.multiply(self.learning_rate, update)

    def predict(self, data):
        pred_target = None
        for tree in self.trees:
            pred_of_tree = tree.predict(data)
            if pred_target is None:
                pred_target = np.zeros_like(pred_of_tree)
            # did max gain, now we are subtracting
            pred_target += np.multiply(self.learning_rate, pred_of_tree)

        # make probability distribution of it
        pred_target = self._softmax(pred_target)
        pred_target = np.argmax(pred_target, axis=1)
        return pred_target

def main(args):
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    clf = GradientBoostedClassifier(trees=args.trees, l2=args.l2,
                                    learning_rate=args.learning_rate, max_depth=args.max_depth,
                                    min_to_split=args.min_to_split)
    clf.fit(train_data, train_target)
    train_predict = clf.predict(train_data)
    test_predict = clf.predict(test_data)
    print(test_predict)
    print(test_target)

    train_accuracies = sklearn.metrics.accuracy_score(train_target, train_predict)
    test_accuracies = sklearn.metrics.accuracy_score(test_target, test_predict)

    return train_accuracies, test_accuracies

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
    parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
    parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")

    parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
    parser.add_argument("--min_to_split", default=2, type=int, help="Minimum number of instances to split")

    parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
    parser.add_argument("--seed", default=57, type=int, help="Random seed")
    # If you add more arguments, ReCodEx will keep them with your default values.

    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(args.trees, 100 * train_accuracy, 100 * test_accuracy))

