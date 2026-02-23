import numpy as np
import random
import matplotlib.pyplot as plt
from .utils import sigmoid, create_binary_vector, get_neighbor


def precompute(marker_scores, sim_scores):
    sig_marker = sigmoid(marker_scores["marker_score"].values)
    sig_sim    = sigmoid(sim_scores.values)
    return sig_marker, sig_sim


def obj(X, nGenes, lambda1, lambda2, lambda3, sig_marker, sig_sim):
    n_selected = X.sum()

    c1 = lambda1 * np.dot(X, sig_marker) / nGenes

    outer = np.outer(X, X)
    np.fill_diagonal(outer, 0)
    c2 = -2 * lambda2 * (np.sum(outer * sig_sim) / 2) / \
         (n_selected * (n_selected - 1) + 1)

    c3 = -lambda3 * n_selected / nGenes

    return c1 + c2 + c3


def _generate_bitstring_groups(n, K):
    """
    Partition the binary search space into K groups and return one
    representative bitstring (as a numpy array) per group.

    Each bitstring is mapped to a group by its integer value relative
    to the total search space size (2^n), so groups cover equal-sized
    regions of the space. Only the first bitstring that falls into each
    empty group is kept — giving K diverse representatives.
    """
    groups    = {}
    total_bits = 2 ** n

    while len(groups) < K:
        b     = np.random.randint(0, 2, size=n)
        b_int = int(''.join(b.astype(str)), 2)
        g     = min(int((b_int / total_bits) * K), K - 1)
        if g not in groups:
            groups[g] = b

    return list(groups.values())


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
    neighbor_mode="standard",   # "standard" or "partitioned"
    n_groups=8,                 # only used when neighbor_mode="partitioned"
):
    """
    neighbor_mode="standard"    : evaluate how_many_neighbors random neighbors per iteration
    neighbor_mode="partitioned" : generate K representative neighbors via bitstring group
                                  partitioning (one per region of the search space)
    """
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

        # ── Generate neighbors ───────────────────────────────────────────────
        if neighbor_mode == "partitioned":
            # K representatives spread across the binary search space
            neighbors = _generate_bitstring_groups(nGenes, n_groups)
        else:
            # Original: how_many_neighbors random perturbations of current solution
            neighbors = [get_neighbor(current_solution, mode, n_flips=n_flips)
                         for _ in range(how_many_neighbors)]

        # ── Explore vs exploit ───────────────────────────────────────────────
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
    neighbor_mode="standard",   # "standard" or "partitioned"
    n_groups=8,                 # only used when neighbor_mode="partitioned"
):
    nGenes = len(marker_scores)

    if mode == 0:
        initial_solution = np.random.randint(2, size=nGenes)
    else:
        initial_solution = create_binary_vector(nGenes, how_many)

    if verbose:
        print("Initial solution:", initial_solution)
        print(f"neighbor_mode={neighbor_mode}" +
              (f"  n_groups={n_groups}" if neighbor_mode == "partitioned"
               else f"  how_many_neighbors={how_many_neighbors}"))

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
        neighbor_mode=neighbor_mode,
        n_groups=n_groups,
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
