#!/usr/bin/env python3
import argparse

import numpy as np
from scipy.stats import multivariate_normal

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", type=str, help="Initialization (random/kmeans++)")
parser.add_argument("--iterations", default=10, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def plot(args, iteration, data, r, means, covs, colors):
    # complex plot method
    # it was given
    import matplotlib.patches
    import matplotlib.pyplot as plt

    if args.plot is not True:
        if not plt.gcf().get_axes(): plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("MoG Initialization" if not iteration else
              "MoG After Initialization {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=r.T @ colors if r is not None else "k")
    for c in range(args.clusters):
        eigvalues, eigvectors = np.linalg.eigh(covs[c])
        ellipse = matplotlib.patches.Ellipse(
            xy=means[c], width=4*np.sqrt(eigvalues[0]), height=4*np.sqrt(eigvalues[1]),
            angle=np.rad2deg(np.arctan2(*eigvectors[0, ::-1])))
        ellipse.set_color(colors[c])
        ellipse.set_alpha(0.3)
        plt.gca().add_artist(ellipse)
    if args.plot is True: plt.show()
    else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate artificial data
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO(kmeans): Initialize `means` with shape [args.clusters, data.shape[1]] as
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster index by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the cluster indices proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.
    if args.init == "random":
        means_ids = generator.choice(len(data), size=args.clusters, replace=False)
        means = data[means_ids]

    elif args.init == "kmeans++":
        means_ids = []
        means_ids.append(generator.randint(len(data)))
        while len(means_ids) < args.clusters:

            square_distances = np.ones(data.shape[0])
            for mean_i in means_ids:
                measure_dist = lambda ins : np.linalg.norm(ins - data[mean_i]) ** 2
                # measure square distances between data and current center
                square_distances *= np.apply_along_axis(measure_dist, 1, data)

            # if it is current center doesnt matter
            # it will take prob 0
            # give higher prob to further away point from current centers
            # ( with an expense of multiplication )
            means_ids.append(generator.choice(len(data), p=square_distances / np.sum(square_distances)))

        means = data[means_ids]

    # TODO: Initialize the cluster covariances in the variable `covs` of
    # shape [args.clusters, data.shape[1], data.shape[1]] as identity matrices.
    covs = np.array([ np.identity(data.shape[1]) for _ in range(args.clusters)])

    # TODO: Initialize the prior distribution `mixing_coefs` as a uniform distribution.
    mixing_coefs = generator.uniform(size=args.clusters)

    if args.plot:
        colors = np.concatenate([[[1,0,0], [0,1,0], [0,0,1]],
                                 np.random.RandomState(4).uniform(size=[args.clusters, 3])], axis=0)[:-3]
        plot(args, 0, data, None, means, covs, colors)

    # Run `args.iterations` of the gaussian mixture fitting algorithm

    def calc_prior_normal_multip(data, covs, means, priors):
        prior_normal_multip = np.zeros((args.clusters, data.shape[0]))

        for k in range(args.clusters):
            prior = priors[k]
            mean = means[k]
            cov = covs[k]
            prob_func = lambda x : prior * multivariate_normal.pdf(x, mean, cov, allow_singular=True)
            prior_normal_multip[k] = np.apply_along_axis(prob_func, 1, data)
        return prior_normal_multip


    losses = []
    for iteration in range(args.iterations):
        # TODO: Evaluate resps. You can use
        # `scipy.stats.multivariate_normal` to calculate PDF of
        # a multivariate Gaussian distribution.
        # Evaluation step of EM

        prior_normal_multip = calc_prior_normal_multip(data, covs, means, mixing_coefs)
        sum_of_prior_normal_multips = np.sum(prior_normal_multip, axis=0)
        print(sum_of_prior_normal_multips)
        resps = prior_normal_multip / sum_of_prior_normal_multips

        # TODO: Update cluster `means`, `covs` and `mixing_coefs`.
        # Maximization step of EM

        for k in range(args.clusters):

            sum_of_resps = np.sum(resps[k,:])
            means[k] = (resps[k, :] @ data) / sum_of_resps

            covs[k] = np.zeros_like(covs[k])
            for i, ins in enumerate(data):
                covs[k] += resps[k, :][i] * ((ins - means[k]) @ (ins - means[k]).T)

            covs[k] = covs[k] / sum_of_resps
            mixing_coefs[k] = sum_of_resps / data.shape[0]

        # TODO: Compute the negative log likelihood of the current model to `loss`.

        prior_normal_multip = calc_prior_normal_multip(data, covs, means, mixing_coefs)

        neg_log_likelihood = - np.log(np.sum(prior_normal_multip, 0))
        loss = np.sum(neg_log_likelihood)

        # Append the current `loss` to `losses`.
        losses.append(loss)

        if args.plot:
            # If you want the plotting code to work, `r` must have shape [args.clusters, data.shape[0]].
            plot(args, 1 + iteration, data, resps, means, covs, colors)

    return losses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    losses = main(args)

    for iteration, loss in enumerate(losses):
        print("Loss after iteration {}: {:.1f}".format(iteration + 1, loss))
