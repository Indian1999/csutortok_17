import numpy as np

# sigmoid: f(x) = 1 / (1 + e^-x)
# Érték készlet: ]0, 1[
# Értelmezési tartomány: valós számok halmaza
# folytonos, szig. mon. növ.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid_deriv: f(x) = x * (1 - x)
def sigmoid_deriv(x):
    return x * (1-x)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - (tanh(x))**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    if x > 0:
        return 1
    if x < 0:
        return 0
    raise ValueError