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


def run_naive_em(X: np.ndarray) -> None:
    """
    Runs the Naive EM Algorithm over the dataset and plots the mixtures
    for K = 3 and seed = 0

    Args:
        X: (n, d) nd-array of Datapoints
    """
    K, seed = 2, 1
    # Initialization
    mixture, post = common.init(X, K, seed)
    mixture, post, ll = naive_em.run(X, mixture, post)

    title = f"K-means for K={K}, seed={seed}, ll={ll}"
    print(title)
    common.plot(X, mixture, post, title)


def test_naive_em(X: np.ndarray) -> None:
    """
    Runs the K-Means Algorithm over the dataset and plots the mixtures 
    for various combinations of K and seed

    Args:
        X: (n, d) nd-array of Datapoints
    """

    for K in range(1, 5):
        for seed in range(0, 5):
            # Initialize the K-Means
            mixture, post = common.init(X, K, seed)
            mixture, post, ll, ll_history = naive_em.run(X, mixture, post)
            title = f"EM for K={K}, seed={seed}, ll={ll}"
            print(title)
            common.em_plot(X, mixture, post, title, ll_history)


test_naive_em(X)
