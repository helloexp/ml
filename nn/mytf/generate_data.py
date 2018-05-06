import matplotlib.pyplot as plt
import numpy as np


def generate_data(w, b, num_excamples=100, noise_sigma=0.1):
    noise = np.random.randn(num_excamples) * noise_sigma
    x = np.linspace(0, 1, num_excamples)
    np.random.shuffle(x)
    y = w * (x + noise) + b
    return x, y


