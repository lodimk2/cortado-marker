import numpy as np
import random
import math
import matplotlib.pyplot as plt
from .utils import (
    sigmoid,
    create_binary_vector,
    get_neighbor
)

def obj(X, nGenes, lambda1, lambda2, lambda3, marker_scores, sim_scores):
    """
    Objective function for stochastic hill climbing.

    Parameters:
    - X (np.array): Binary vector
    - nGenes (int): Number of genes
    - lambda1 (float): Weight for marker gene score
    - lambda2 (float): Weight for similarity score
    - lambda3 (float): Weight for gene set size
    - marker_scores (pd.DataFrame): Marker gene scores
    - sim_scores (pd.DataFrame): Gene correlation scores

    Returns:
    - float: Objective function value
    """
    c1 = lambda1 * (np.sum([X[g] * sigmoid(marker_scores.iloc[g]['marker_score']) for g in range(nGenes)]) / nGenes)

    c2 = -2 * lambda2 * (np.sum([X[g1] * X[g2] * sigmoid(sim_scores.iloc[g1, g2])
                                 for g1 in range(nGenes - 1)
                                 for g2 in range(g1 + 1, nGenes)]) /
                         (np.sum([X[g] for g in range(nGenes)]) * (np.sum([X[g] for g in range(nGenes)]) - 1) + 1))

    c3 = -lambda3 * (np.sum([X[g] for g in range(nGenes)]) / nGenes)

    return c1 + c2 + c3

def stochastic_hill_climbing_adaptive(f, initial_solution, max_iterations, gamma, idle_limit, 
                                      how_many_neighbors, nGenes, lambda1, lambda2, lambda3, marker_scores, sim_scores, mode):
    """
    Stochastic hill climbing algorithm with adaptive exploration reduction.

    Parameters:
    - f (function): Objective function
    - initial_solution (np.array): Initial binary vector
    - max_iterations (int): Maximum number of iterations
    - gamma (float): Exploration rate
    - idle_limit (int): Idle limit before stopping
    - how_many_neighbors (int): Number of neighbors to consider
    - nGenes (int): Number of genes
    - lambda1 (float): Weight for marker gene score
    - lambda2 (float): Weight for similarity score
    - lambda3 (float): Weight for gene set size
    - marker_scores (pd.DataFrame): Marker gene scores
    - sim_scores (pd.DataFrame): Gene correlation scores
    - mode (int): Mode for generating initial solution

    Returns:
    - best_solution (np.array): Best binary vector
    - best_value (float): Best objective function value
    """
    current_solution = initial_solution
    current_value = f(current_solution, nGenes, lambda1, lambda2, lambda3, marker_scores, sim_scores)

    t = 0
    idle_steps = 0
    best_solution = current_solution
    best_value = current_value

    while t < max_iterations and idle_steps < idle_limit:

        exploration_rate = gamma ** t
        print(f'At time {t}, the best value is {best_value}, with an exploration rate of {exploration_rate}.')

        Log.append(best_value)
        t += 1

        neighbors = [get_neighbor(current_solution, mode) for _ in range(how_many_neighbors)]

        if random.uniform(0, 1) < exploration_rate:
            ind = random.choice([i for i in range(how_many_neighbors)])
            current_solution, current_value = neighbors[ind], f(neighbors[ind], nGenes, lambda1, lambda2, lambda3, marker_scores, sim_scores)
            continue

        neighbor_values = [f(neighbor, nGenes, lambda1, lambda2, lambda3, marker_scores, sim_scores) for neighbor in neighbors]

        better_neighbors = [(neighbors[i], neighbor_values[i]) for i in range(len(neighbors)) if
                            neighbor_values[i] > current_value]

        if better_neighbors:
            ind = random.choice([i for i in range(how_many_neighbors)])
            current_solution, current_value = neighbors[ind], f(neighbors[ind], nGenes, lambda1, lambda2, lambda3, marker_scores, sim_scores)

            if current_value > best_value:
                best_solution, best_value = current_solution, current_value
            idle_steps = 0
        else:
            idle_steps += 1

    return best_solution, best_value
def run_stochastic_hill_climbing(marker_scores, filtered_corr_matrix, how_many=25, max_iterations=100, gamma=0.95, 
                                 idle_limit=10, lambda1=0.7, lambda2=0.2, lambda3=0.1, mode=1, plot_filename='cost_plot.png'):
    """
    Run stochastic hill climbing algorithm.

    Parameters:
    - marker_scores (pd.DataFrame): Marker gene scores
    - filtered_corr_matrix (pd.DataFrame): Filtered gene correlation scores
    - how_many (int): Number of 1s in the binary vector
    - max_iterations (int): Maximum number of iterations
    - gamma (float): Exploration rate
    - idle_limit (int): Idle limit before stopping
    - lambda1 (float): Weight for marker gene score
    - lambda2 (float): Weight for similarity score
    - lambda3 (float): Weight for gene set size
    - mode (int): Mode for generating initial solution
    - plot_filename (str): Filename for the cost plot

    Returns:
    - best_solution (np.array): Best binary vector
    - best_value (float): Best objective function value
    """
    nGenes = len(marker_scores)
    
    # Initialize solution
    if mode == 0:
        initial_solution = np.random.randint(2, size=nGenes)
    else:
        initial_solution = create_binary_vector(nGenes, how_many)

    print("Initial solution:", initial_solution)
    
    # Initialize logging for the cost function values
    global Log
    Log = []

    # Run the hill-climbing algorithm
    best_solution, best_value = stochastic_hill_climbing_adaptive(obj, initial_solution, max_iterations, gamma, idle_limit, 
                                                                  10, nGenes, lambda1, lambda2, lambda3, marker_scores, filtered_corr_matrix, mode)

    # Plot the cost function and save the plot
    plt.plot([i for i in range(len(Log))], Log)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations')
    plt.savefig(plot_filename)
    plt.close()

    print(f"Cost plot saved as {plot_filename}")
    print("Best Solution:", best_solution)
    print("Best Value:", best_value)

    return best_solution, best_value