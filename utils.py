import os
import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional, Callable, Dict

from query import (
    get_starting_points,
    get_optimizer_number,
    get_learning_rate,
    get_function_number
)

from loss_functions import (
    matyas,
    himmelblau,
)


def plot_graph(history: Dict[str, float], loss_fn: Callable,
               optimizer: str) -> None:
    """Renders and saves an animation of the data passed in the `history`
    parameter.

    Parameters
    ----------
    history : dict[str, float]
        The values of the weights after every iteration of gradient descent.

    loss_fn : callable
        The loss function upon which the weights were trained.

    optimizer: str
        The name of the optimizer used for gradient descent.
    """

    def animate(i, dataset, line, c_line):
        line.set_data(dataset[0:2, :i])
        line.set_3d_properties(dataset[2, :i])
        c_line.set_data(dataset[0:2, :i])
        return line, c_line

    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    plt.style.use('seaborn')

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=30, azim=130)

    if loss_fn.__class__.__name__ == 'matyas':
        ax1.view_init(elev=15, azim=75)

    x = np.linspace(-5.2, 5.2, 25)
    y = np.linspace(-5.2, 5.2, 25)
    X, Y = np.meshgrid(x, y)  # all possible combinations of x and y
    Z = loss_fn([X, Y])
    ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)

    levels = np.linspace(0, 500, 30)  # contours from 0 to 500, of 30 parts
    ax2 = fig.add_subplot(122)
    ax2.contourf(X, Y, Z, levels, cmap='jet', alpha=0.5)

    x_history = np.array([i[0] for i in history['weights']])
    y_history = np.array([i[1] for i in history['weights']])
    z_history = np.array(history['loss'])
    dataset = np.array([x_history, y_history, z_history])

    total_iter = len(x_history) - 1
    n_frames = total_iter + 1
    interval = 5 * 1000 / n_frames
    fps = (1 / (interval / 1000))

    line = ax1.plot(dataset[0], dataset[1], dataset[2], label='optimization',
                    c='r', marker='.', alpha=0.4)[0]
    ax1.set_title(f'{loss_fn.__name__} using {optimizer}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x, y)')
    ax1.legend()

    c_line = ax2.plot(dataset[0], dataset[1], label='optimization', c='r',
                      marker='.', alpha=0.4)[0]

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False,
                                  interval=interval, repeat=True,
                                  fargs=(dataset, line, c_line))

    print("Saving animation...")
    ani.save(f"visualizations/{loss_fn.__name__}_{optimizer}.mp4", fps=fps)
    print("Animation saved!")

    plt.show()


def get_loss_fn() -> Callable:
    """Returns a callable function that is to be minimized.

    Returns
    -------
    Callable
        The function for which the gradients will be calculated.
    """

    func_num = get_function_number()

    if func_num == 1:
        loss_fn = matyas
    elif func_num == 2:
        loss_fn = himmelblau

    return loss_fn


def get_optimizer(weights: torch.Tensor) -> torch.optim:
    """Returns a pytorch optimizer, based on user selection.

    Parameters
    ----------
    weights : torch.Tensor
        The variables for which gradient is to be computed.

    Returns
    -------
    torch.optim
        A specific pytorch optimizer based on user preference.
    """

    opt_num = get_optimizer_number()
    lr = get_learning_rate()

    if opt_num == 1:
        opt = torch.optim.SGD([weights], lr=lr)
    elif opt_num == 2:
        opt = torch.optim.Adam([weights], lr=lr)
    elif opt_num == 3:
        opt = torch.optim.RMSprop([weights], lr=lr)

    return opt


def get_initial_weights(random: bool, device: torch.device) -> torch.Tensor:
    """Returns weights that are to be used as initial weights.

    Parameters
    ----------
    random : bool
        A boolean value indicating whether the random weights must be returned.

    device : torch.device
        Device used for the computation, either CPU or GPU.

    Returns
    -------
    torch.Tensor

    Notes
    -----
    If `random`, then weights are drawn from a normal distribution.
    Else, user input is taken as weights.
    """

    if random:
        return torch.normal(mean=0.0, std=2.23, size=(2,), dtype=torch.float64,
                            requires_grad=True, device=device)

    x, y = get_starting_points()
    return torch.tensor([x, y], dtype=torch.float64, requires_grad=True,
                        device=device)


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
