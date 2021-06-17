import numpy as np
from numpy import ndarray
from typing import List

from base import Operation, WeightMultiply, BiasAdd, Conv2D_Op
from utils import assert_same_shape, permute_data
from activations import Sigmoid
from base import ParamOperation


class Layer(object):
    '''
    A layer of neurons in a Neural Network
    '''

    def __init__(self, neurons: int):
        '''
        Initialize the layer with 'neurons' neurons in the layer
        '''

        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: ndarray) -> None:
        '''
        The _setup_layer function needs to be implemented for each layer
        '''

        raise NotImplementedError()

    def forward(self, input_: ndarray, inference: bool = False) -> ndarray:
        '''
        Passes the input forward through a series of operations
        '''

        if self.first:
            self._setup_layer(input_)
            self.first = False
        
        self.input_ = input_

        for operation in self.operations:
            # print(operation)
            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    
    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Passes output_grad backward through a series of operations and checks the appropriate shapes
        '''

        assert_same_shape(self.output, output_grad)
        
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad) 

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self):
        '''
        Extracts a _param_grads from a layers operations
        '''
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        '''
        Extracts the _params from a layer's operations.
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

class Dense(Layer):
    '''
    A fully connected layer
    '''

    def __init__(self, neurons: int, activation: Operation = Sigmoid(), dropout: float = 1.0, weight_init: str = 'standard'):
        super().__init__(neurons)
        self.dropout = dropout
        self.activation = activation
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> ndarray:
        '''
        Defines the operations of a fully connected layer.
        '''

        if self.seed:
            np.random.seed(self.seed)

        num_in = input_.shape[1]
        if self.weight_init == 'glorot':
            scale = 2 / (num_in + self.neurons)
        
        else:
            scale = 1.0

        
        self.params = []

        # weights
        self.params.append(np.random.normal(loc=0, scale=scale, size=(num_in, self.neurons)))

        # bias
        self.params.append(np.random.normal(loc=0, scale=scale, size=(1, self.neurons)))

        self.operations = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None



class Dropout(Operation):
    '''
    Dropout Layer

    '''
    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference: bool) -> ndarray:
        if inference:
            return self._input * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self._input.shape)
            return self._input * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask


class Flatten(Operation):
    '''
    Flatten layer
    '''
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return self._input.reshape(self._input.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad.reshape(self._input.shape)


class Conv2D(Layer):
    '''
    Convolutional Layer
    '''
    def __init__(self, out_channels: int, 
                        param_size: int, 
                        dropout: float = 1.0,
                        weight_init: str = 'normal',
                        activation: Operation = Sigmoid(), 
                        flatten: bool = False):
        super().__init__(out_channels)
        self.out_channels = out_channels
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init

    def _setup_layer(self, input_: ndarray) -> ndarray:
        
        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == 'glorot':
            scale = 2 / (in_channels + self.out_channels)
        else:
            scale = 1.0
        
        conv_param = np.random.normal(loc = 0, scale=scale, 
                                    size=(in_channels, self.out_channels, self.param_size, self.param_size))
        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)
        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))
        return None
