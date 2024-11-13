import numpy as np


def sigmoid(x):
    if x > 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        a = np.exp(x)
        return a / (1.0 + a)


def dsigmoid_arr(z):
    return z * (1 - z)
