import numpy as np
import random
import matplotlib.pyplot as plt
from .utils import sigmoid, create_binary_vector, get_neighbor


def precompute(marker_scores, sim_scores):
    """
    Pre-compute sigmoid arrays once before the hill-climbing loop.
    Avoids redundant pandas indexing and sigmoid calls inside obj.
    """
    sig_marker = sigmoid(marker_scores["marker_score"].values)  # (nGenes,)
    sig_sim    = sigmoid(sim_scores.values)                      # (nGenes, nGenes)
    return sig_marker, sig_sim


def obj(X, nGenes, lambda1, lambda2, lambda3, sig_marker, sig_sim):
    """
    Vectorized objective function.

    Parameters:
    - X (np.array): Binary vector
    - nGenes (int): Number of genes
    - lambda1 (float): Weight for marker gene score
    - lambda2 (float): Weight for similarity score
    - lambda3 (float): Weight for gene set size
    - sig_marker (np.array): Pre-computed sigmoid of marker scores
    - sig_sim (np.array): Pre-computed sigmoid of sim scores

    Returns:
    - float: Objective function value
    """
    n_selected = X.sum()

    c1 = lambda1 * np.dot(X, sig_marker) / nGenes

    outer = np.outer(X, X)
    np.fill_diagonal(outer, 0)
    c2 = -2 * lambda2 * (np.sum(outer * sig_sim) / 2) / \
         (n_selected * (n_selected - 1) + 1)

    c3 = -lambda3 * n_selected / nGenes

    return c1 + c2 + c3


def stochastic_hill_climbing_adaptive(
    f,
    initial_solution,
    max_iterations,
    gamma,
    idle_limit,
    how_many_neighbors,
    nGenes,
    lambda1,
    lambda2,
    lambda3,
    marker_scores,
    sim_scores,
    mode,
    n_flips=1,
    verbose=False,
):
    sig_marker, sig_sim = precompute(marker_scores, sim_scores)

    current_solution = initial_solution.copy()
    current_value    = f(current_solution, nGenes, lambda1, lambda2, lambda3,
                         sig_marker, sig_sim)
    best_solution    = current_solution.copy()
    best_value       = current_value
    log              = []
    t                = 0
    idle_steps       = 0

    while t < max_iterations and idle_steps < idle_limit:
        exploration_rate = gamma ** t

        if verbose:
            print(f"t={t}  best={best_value:.6f}  exploration={exploration_rate:.4f}")

        log.append(best_value)
        t += 1

        neighbors = [get_neighbor(current_solution, mode, n_flips=n_flips)
                     for _ in range(how_many_neighbors)]

        if random.uniform(0, 1) < exploration_rate:
            idx              = random.randrange(len(neighbors))
            current_solution = neighbors[idx]
            current_value    = f(current_solution, nGenes, lambda1, lambda2,
                                 lambda3, sig_marker, sig_sim)
            continue

        neighbor_values = [f(n, nGenes, lambda1, lambda2, lambda3,
                             sig_marker, sig_sim) for n in neighbors]
        better = [i for i in range(len(neighbors))
                  if neighbor_values[i] > current_value]

        if better:
            idx              = random.choice(better)
            current_solution = neighbors[idx]
            current_value    = neighbor_values[idx]
            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value    = current_value
            idle_steps = 0
        else:
            idle_steps += 1

    return best_solution, best_value, log


def run_stochastic_hill_climbing(
    marker_scores,
    filtered_corr_matrix,
    how_many=25,
    max_iterations=100,
    gamma=0.95,
    idle_limit=10,
    how_many_neighbors=10,
    n_flips=1,
    lambda1=0.7,
    lambda2=0.2,
    lambda3=0.1,
    mode=1,
    plot_filename='cost_plot.png',
    verbose=False,
):
    nGenes = len(marker_scores)

    if mode == 0:
        initial_solution = np.random.randint(2, size=nGenes)
    else:
        initial_solution = create_binary_vector(nGenes, how_many)

    if verbose:
        print("Initial solution:", initial_solution)

    best_solution, best_value, log = stochastic_hill_climbing_adaptive(
        obj,
        initial_solution,
        max_iterations,
        gamma,
        idle_limit,
        how_many_neighbors,
        nGenes,
        lambda1,
        lambda2,
        lambda3,
        marker_scores,
        filtered_corr_matrix,
        mode,
        n_flips=n_flips,
        verbose=verbose,
    )

    if plot_filename:
        plt.plot(range(len(log)), log)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function Over Iterations')
        plt.savefig(plot_filename)
        plt.close()

    if verbose:
        print("Best Solution:", best_solution)
        print("Best Value:", best_value)

    return best_solution, best_value
