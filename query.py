import numpy as np
from typing import Tuple


def get_iteration_number() -> int:
    ui = input("Enter the number of iterations for gradient descent, leave blank for 100: ")
    if ui == '':
        return 100

    return int(ui)


def get_function_number() -> int:
    print("\nSelect the function to minimize:")
    print("\t1. Matyas function")
    print("\t\tGlobal Minima: (0, 0) = 0")

    print("\t2. Himmelblau's function")
    print("\t\tLocal Minima: (3, 2) = 0")
    print("\t\tLocal Minima: (-2.805118, 3.131312) = 0")
    print("\t\tLocal Minima: (-3.779310, -3.283186) = 0")
    print("\t\tLocal Minima: (3.584428, -1.848126) = 0")

    ui = int(input("\nEnter the number of the function to minimize: "))
    return ui


def get_learning_rate() -> float:
    ui = input("Enter the learning rate to be used, leave blank for 1e-1: ")
    if ui == '':
        return 1e-1

    return float(ui)


def get_optimizer_number() -> int:
    print("\nSelect the optimizer to use:")
    print("\t1. Stochastic Gradient Descent (SGD)")
    print("\t2. Adaptive Moment Estimation (Adam)")
    print("\t3. Root Mean Squared Propagation (RMSprop)")

    ui = int(input("\nEnter the number for the optimizer to use: "))
    return ui


def get_starting_points() -> Tuple[float, float]:
    x = float(input("Enter the starting x coordinate: "))
    y = float(input("Enter the starting y coordinate: "))

    return (x, y)


def is_random_initialization() -> bool:
    ui = input("Do you want the weights to be initialized randomly? [y/n]: ")
    return ui.lower() != 'n'


def _is_random_seed() -> bool:
    ui = input("Do you want to initialize the seed randomly? [y/n]: ")
    return ui.lower() != 'n'


def _get_seed_value() -> int:
    seed = input("Enter the value for seed, leave blank for 42: ")

    if seed == '':
        return 42
    return int(seed)


def get_seed() -> int:
    if _is_random_seed():
        return np.random.randint(1, 101)

    return _get_seed_value()
