from utils import (
    set_pytorch_seed,
    get_pytorch_device,
    setup_logger,
    get_initial_weights,
    get_optimizer,
    get_loss_fn
)

from query import (
    is_random_initialization,
    get_seed,
    get_iteration_number
)

logger = setup_logger("Stream Logger")
random = is_random_initialization()

if random:
    seed = get_seed()
    set_pytorch_seed(seed)
    logger.debug(f"Seed set as {seed} for reproducibility.")

device = get_pytorch_device()
logger.debug(f"Using device: {device}")

loss_fn = get_loss_fn()
weights = get_initial_weights(random, device)
logger.info(f"Initial Weights: {weights.cpu().detach().numpy()}")

optimizer = get_optimizer(weights)
n_iteration = get_iteration_number()

for i in range(1, n_iteration + 1):
    loss = loss_fn(weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i % (n_iteration / 10) == 0 or i % n_iteration == 0):
        logger.info(f"Iteration {i}, Weight: {weights.cpu().detach().numpy()}")

logger.info(f"Value: {loss_fn(weights)} at ({weights[0]}, {weights[1]})")
