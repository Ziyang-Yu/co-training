
import random
import numpy as np
import torch
def seed(seed_val):
    # import tensorflow as tf
    # tf.random.set_seed(seed_value)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)