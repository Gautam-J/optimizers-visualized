import numpy as np

from typing import Tuple


def get_starting_points() -> Tuple[float, float]:
    x = float(input("Enter the starting x coordinate: "))
    y = float(input("Enter the starting y coordinate: "))

    return (x, y)


def is_random_initialization():
    ui = input("Do you want the weights to be initialized randomly? [y/n]: ")
    if ui.lower() == 'n':
        return False

    return True


def _is_random_seed():
    ui = input("Do you want to initialize the seed randomly? [y/n]: ")
    if ui.lower() == 'n':
        return False

    return True


def _get_seed_value():
    seed = input("Enter the value for seed, leave blank for 42: ")

    if seed == '':
        return 42
    return int(seed)


def get_seed():
    if _is_random_seed():
        return np.random.randint(1, 101)

    return _get_seed_value()
