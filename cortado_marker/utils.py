import math
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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


def get_neighbor(solution, mode, n_flips=1):
    """
    Get a neighbor of a given binary vector.

    Parameters
    ----------
    solution : np.array
        Binary vector representing the current selection.
    mode : int
        0 – free flipping: flip n_flips random bits with no constraint.
        1 – balance-preserving: each flip is immediately compensated by
            flipping a second bit of the same new value, keeping the total
            number of selected genes approximately stable.
    n_flips : int
        Number of bit-flip operations to perform (default 1, matching
        original behaviour).

    Returns
    -------
    np.array
        Neighbour binary vector.
    """
    neighbor = solution.copy()

    for _ in range(n_flips):
        flip_index = random.randint(0, len(neighbor) - 1)
        neighbor[flip_index] = 1 - neighbor[flip_index]   # primary flip

        if mode == 1:
            new_bit = neighbor[flip_index]
            # Compensating flip: pick another bit with the same new value
            # (excludes the just-flipped index to avoid immediately undoing it)
            candidates = [i for i in range(len(neighbor))
                          if neighbor[i] == new_bit and i != flip_index]
            if candidates:
                compensate_index = random.choice(candidates)
                neighbor[compensate_index] = 1 - neighbor[compensate_index]

    return neighbor
