import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


def run_kmeans(X: np.ndarray) -> None:
    """
    Runs the K-Means Algorithm over the dataset and plots the mixtures 
    for various combinations of K and seed

    Args:
        X: (n, d) nd-array of Datapoints
    """

    for K in range(1, 7):
        min_cost = None
        best_seed = None
        for seed in range(0, 5):
            # Initialize the K-Means
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = f"K-means for K={K}, seed={best_seed}, cost={min_cost}"
        print(title)
        common.plot(X, mixture, post, title)


run_kmeans(X)
