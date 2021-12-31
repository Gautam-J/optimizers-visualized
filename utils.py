import os
import torch
import random
import logging
import numpy as np

from typing import Optional

from query import get_starting_points


def get_initial_weights(random: bool) -> torch.Tensor:
    """Returns weights that are to be used as initial weights.

    Parameters
    ----------
    random : bool
        A boolean value indicating whether the random weights must be returned.

    Returns
    -------
    torch.Tensor

    Notes
    -----
    If `random`, then weights are drawn from a normal distribution.
    Else, user input is taken as weights.
    """

    if random:
        return torch.normal(mean=0.0, std=2.45, size=(2,), dtype=torch.float64,
                            requires_grad=True)

    x, y = get_starting_points()
    return torch.tensor([x, y], dtype=torch.float64, requires_grad=True)


def set_pytorch_seed(seed: Optional[int] = 42) -> None:
    """Sets the random seed to be used for all computations.

    Parameters
    ----------
    seed : int, default=42
        The value of the seed to be used.

    Notes
    -----
    Function sets seed for the following modules: `random`, `numpy`, `torch`
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_pytorch_device() -> torch.device:
    """Checks if a CUDA enabled GPU is available, and returns the
    approriate device, either CPU or GPU.

    Returns
    -------
    device : torch.device
    """

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    return device


def setup_logger(name: str,
                 formatter: Optional[logging.Formatter] = None,
                 log_file: Optional[str] = None,
                 level: Optional[int] = logging.DEBUG) -> logging.Logger:
    """Set up a python logger to log results.

    Parameters
    ----------
    name : str
        Name of the logger.

    formatter : logging.Formatter, default=None
        A custom formatter for the logger to output. If None, a default
        formatter of format `"%Y-%m-%d %H:%M:%S LEVEL MESSAGE"` is used.

    log_file : str, default=None
        File path to record logs. Must end with a readable extension. If None,
        the logs are not logged in any file, and are logged only to `stdout`.

    level : int, default=10 (logging.DEBUG)
        Base level to log. Any level lower than this level will not be logged.

    Returns
    -------
    logger : logging.Logger
        A logger with formatters and handlers attached to it.

    Notes
    -----
    If passing a directory name along with the log file, make sure the
    directory exists.
    """

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    if log_file is not None:
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger
