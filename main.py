from utils import (
    set_pytorch_seed,
    get_pytorch_device,
    setup_logger,
    get_initial_weights
)

from query import (
    is_random_initialization,
    get_seed
)

logger = setup_logger("Stream Logger")
random = is_random_initialization()

if random:
    seed = get_seed()
    set_pytorch_seed(seed)
    logger.info(f"Seed set as {seed} for reproducibility.")

device = get_pytorch_device()
logger.info(f"Using device: {device}")

weights = get_initial_weights(random).to(device)
logger.info(f"Initial Weights: {weights.cpu().detach().numpy()}")
