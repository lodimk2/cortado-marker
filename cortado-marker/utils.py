import math
import numpy as np
import random
def sigmoid(x):
    """
    Sigmoid function.

    Parameters:
    - x (float): Input value

    Returns:
    - float: Sigmoid output
    """
    return 1 / (1 + math.exp(-x))

def create_binary_vector(length, num_ones):
    """
    Create a binary vector with a specified number of 1s.

    Parameters:
    - length (int): Length of the binary vector
    - num_ones (int): Number of 1s in the binary vector

    Returns:
    - np.array: Binary vector
    """
    binary_vector = np.array([1] * num_ones + [0] * (length - num_ones))
    np.random.shuffle(binary_vector)
    return binary_vector


def get_neighbor(solution, mode):
    """
    Get a neighbor of a given binary vector.

    Parameters:
    - solution (np.array): Binary vector
    - mode (int): Mode for generating neighbor

    Returns:
    - np.array: Neighbor binary vector
    """
    neighbor = solution.copy()
    flip_index = random.randint(0, len(solution) - 1)
    neighbor[flip_index] = 1 - neighbor[flip_index]  # Flip the bit

    new_Bit = neighbor[flip_index]
    if mode == 1:
        indices = [i for i in range(len(neighbor)) if neighbor[i] == new_Bit]
        flip_index = random.choice(indices)
        neighbor[flip_index] = 1 - neighbor[flip_index]

    return neighbor