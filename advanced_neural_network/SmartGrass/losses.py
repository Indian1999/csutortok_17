import numpy as np

# Mean Squared Error (Átlagos négyzetes hiba)
def mse(y_expected, y_predicted):
    return np.mean((y_expected - y_predicted) ** 2)

def mse_deriv(y_expected, y_predicted):
    return 2 * (y_expected - y_predicted) / y_predicted.size

def categorical_cross_entropy(y_expected, y_predicted):
    return - np.sum(y_expected * np.log(y_predicted))

def categorical_cross_entropy_deriv(y_expected, y_predicted):
    return - y_expected/y_predicted