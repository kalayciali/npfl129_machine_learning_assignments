#!/usr/bin/env python3
import argparse

from recordclass import recordclass
import math

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection



Node = recordclass('Node', ['left', 'right', 'feat_split', 'counts', 'criterion', 'depth'])

class DecisionTreeClassifier():

    def __init__(self, criterion="gini", max_depth=None, 
                 max_leaves=None, min_to_split=2):

        self.criterion = self._gini if criterion == "gini" else self._entropy
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_to_split = min_to_split
        self.feature_splits = None
        self.root = None

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

    def _gini(self, counts):
        data_size = sum(counts)
        probs = counts / data_size
        return data_size * sum([p * (1 - p) for p in probs])

    def _entropy(self, counts):
        data_size = sum(counts)
        probs = counts / data_size
        return - data_size * sum([p * math.log(p)  for p in probs])

    def _calc_counts(self, target):
        _, counts = np.unique(target, return_counts=True)
        return counts

    def _split_data(self, data, target, more_l_r=False):
        best_criterion = np.inf
        best_l = None
        best_r = None

        root_counts = self._calc_counts(target)
        root_criterion = self.criterion(root_counts)

        for feat, splits_to_consider in enumerate(self.feature_splits):
            for split in splits_to_consider:

                l_indices = (data[:, feat] <= split).nonzero()[0]
                l_counts = self._calc_counts(target[l_indices])
                l_criterion = self.criterion(l_counts)

                r_indices = (data[:, feat] > split).nonzero()[0]
                r_counts = self._calc_counts(target[r_indices])
                r_criterion = self.criterion(r_counts)

                criterion_if_split = l_criterion + r_criterion - root_criterion

                if criterion_if_split < best_criterion:
                    best_criterion = criterion_if_split
                    best_split = (feat, split)

                    l_data, l_target = data[l_indices], target[l_indices]
                    r_data, r_target = data[r_indices], target[r_indices]

                    if more_l_r:
                        best_l = (l_data, l_target, l_counts, l_criterion)
                        best_r = (r_data, r_target, r_counts, r_criterion)
                    else:
                        best_l = (l_data, l_target)
                        best_r = (r_data, r_target)



        return best_l, best_r, best_split, best_criterion
    
    def _construct_tree(self, data, target, depth = 0):
        root_counts = self._calc_counts(target)
        root_criterion = self.criterion(root_counts)

        # - Allow splitting a node only if:
        #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
        #     depth of the root node is zero;
        #   - there are at least `args.min_to_split` corresponding instances;
        #   - the criterion value is not zero.

        # - When `args.max_leaves` is None, use recursive (l descendants first, then
        #   r descendants) approach, splitting every node if the constraints are valid.
        #   Otherwise (when `args.max_leaves` is not None), always split a node where the
        #   constraints are valid and the overall criterion value (c_l + c_r - c_node)
        #   decreases the most. If there are several such nodes, choose the one
        #   which was created sooner (a l child is considered to be created
        #   before a r child).
            
        if not self.max_leaves:
            # recursive implementation

            if (root_criterion > 0 and data.shape[0] >= self.min_to_split):
                if (self.max_depth and depth < self.max_depth) or not self.max_depth:
                    best_l, best_r, best_split, _ = self._split_data(data, target)

                    tree = Node(self._construct_tree(*best_l, depth + 1),
                                self._construct_tree(*best_r, depth + 1),
                                best_split, root_counts, root_criterion, depth)
                    return tree

                else:
                    leaf = Node(None, None, None, root_counts, root_criterion, depth)
                    return leaf
            else:
                leaf = Node(None, None, None, root_counts, root_criterion, depth)
                return leaf

        else:
            # breadth-first search

            tree = Node(None, None, None, root_counts, root_criterion, depth)

            current_leaves = [(data, target, tree), ]

            while len(current_leaves) < self.max_leaves:

                best_i = None
                best_criterion, best_split = np.inf, None

                splitted = False

                for i, leaf in enumerate(current_leaves):

                    if (leaf[2].criterion > 0 and leaf[0].shape[0] >= self.min_to_split):
                        if (self.max_depth and leaf[2].depth < self.max_depth) or not self.max_depth:

                            left, right, split, criterion = self._split_data(leaf[0], leaf[1], more_l_r=True)

                            if criterion < best_criterion:
                                splitted = True
                                best_criterion = criterion
                                best_i = i
                                best_split = split
                                best_l = left
                                best_r = right

                        else:
                            continue
                    else:
                        continue

                if splitted:

                    splitted_node = current_leaves[best_i][2]

                    depth = splitted_node.depth

                    l_data, l_target, l_counts, l_criterion = best_l
                    r_data, r_target, r_counts, r_criterion = best_r


                    del current_leaves[best_i]
                    print(splitted_node)

                    node_left = Node(None, None, None, l_counts, l_criterion, depth + 1)
                    splitted_node.left = node_left
                    current_leaves.append((l_data, l_target, node_left))

                    node_right = Node(None, None, None, r_counts, r_criterion, depth + 1)
                    splitted_node.right = node_right
                    current_leaves.append((r_data, r_target, node_right))

                else:
                    break

            return tree

    def fit(self, data, target):
        self.feature_splits = self._gen_feature_splits(data)
        self.root = self._construct_tree(data, target)
        return self

    def _predict_one(self, ins):
        current_node = self.root

        while current_node.feat_split:
            feat_i, val = current_node.feat_split
            if ins[feat_i] <= val:
                current_node = current_node.left
            else:
                current_node = current_node.right

        predict = np.argmax(current_node.counts)
        return predict

    def predict(self, data):
        predictions = np.apply_along_axis(self._predict_one, 1, data)
        return predictions
            

def main(args):
    # Use the wine dataset
    # targets are 0, 1, 2
    # 13 features
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    tree = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth,
                                  max_leaves=args.max_leaves, min_to_split=args.min_to_split)

    tree.fit(train_data, train_target)
    train_pred = tree.predict(train_data)
    test_pred = tree.predict(test_data)


    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_pred)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_pred)

    return train_accuracy, test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
    parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
    parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
    parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
    parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
    # If you add more arguments, ReCodEx will keep them with your default values.

    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))

