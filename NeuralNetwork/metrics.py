import numpy as np
from numpy import ndarray

def mean_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Compute Mean Absolute Error
    '''
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: ndarray, y_pred: ndarray, squared: bool = False) -> float:
    '''
    Compute the RMSE 
    '''
    if squared:
        return np.mean(np.power(y_true - y_pred, 2))
    else:
        return np.sqrt(np.mean(np.power(y_true - y_pred, 2))) 