import numpy as np

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)  # Clip input to avoid overflow
    return 1 / (1 + np.exp(-x_clipped))