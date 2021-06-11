from copy import deepcopy
import numpy as np
from numpy import ndarray

from typing import List, Tuple

from utils import assert_same_shape, permute_data

class Operation(object):
    '''
    Base class for an operation in Neural Network
    '''
    def __init__(self):
        pass

    def forward(self, input_: ndarray, inference: bool = False):
        '''
        Stores input in self._input instance variable and calls the self._output() function
        '''
        self._input = input_
        self.output = self._output(inference)

        return self.output
    
    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls the self._input_grad() function and checks that the appropriate shapes match.
        '''

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self._input, self.input_grad)

        return self.input_grad

    def _output(self, inference: bool) -> ndarray:
        '''
        This method needs to be implemented for each operation
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad:ndarray) -> ndarray:
        '''
        This method needs to be implemented for each operation
        '''
        raise NotImplementedError()


class ParamOperation(Operation):
    '''
    An operation with Parameters
    '''
    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls self._input_grad and self._param_grad and check appropriate shapes.
        '''
        
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self._input, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Every subclass of ParamOperation must implement _param_grad
        '''
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for a neural network
    '''
    
    def __init__(self, W: ndarray):
        super().__init__(W)
    
    def _output(self, inference: bool) -> ndarray:
        '''
        Compute output
        '''
        return np.dot(self._input, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient
        '''

        return np.dot(output_grad, np.transpose(self.param, (1, 0)))
        
    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute parameter gradient
        '''

        return np.dot(np.transpose(self._input, (1, 0)), output_grad)

 
class BiasAdd(ParamOperation):
    '''
    Compute Bias Addition
    '''
    def __init__(self, B: ndarray):
        assert B.shape[0] == 1
        super().__init__(B)
    
    def _output(self, inference: bool) -> ndarray:
        '''
        Compute Output
        '''
        return self._input + self.param 

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient
        '''
        return np.ones_like(self._input) * output_grad 

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute parameter gradient
        '''
        
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
