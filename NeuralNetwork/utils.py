from numpy import ndarray
import numpy as np

from scipy.special import logsumexp

def assert_same_shape(array: ndarray, array_grad: ndarray):
    '''
    Check the shape of arrays and its corresponding gradients
    '''
    assert array.shape == array_grad.shape, f''' Two ndarrays should have the same shape, 
    instead, first ndarray's shape is {array.shape} and second ndarray's shape is {array_grad.shape}. '''
    return None


def permute_data(X, y):
    '''
    Shuffle the data
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def softmax(x: ndarray, axis=None):
    '''
    Perform Softmax Transformation
    '''
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def one_hot_encode(y: ndarray, n_classes: int) -> ndarray:
    '''
    Perform One hot encoding
    '''
    res = np.eye(n_classes)[np.array(y).reshape(-1)]
    return res.reshape(list(y.shape)+[n_classes])

def assert_dim(t: ndarray,
               dim: ndarray):
    assert len(t.shape) == dim, \
    '''
    Tensor expected to have dimension {0}, instead has dimension {1}
    '''.format(dim, len(t.shape))
    return None