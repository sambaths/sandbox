import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

