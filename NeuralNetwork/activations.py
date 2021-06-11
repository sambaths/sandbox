
from numpy import ndarray
import numpy as np

from base import Operation, ParamOperation

class Sigmoid(Operation):
    '''
    Sigmoid Activation Function
    '''
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        '''
        Compute output
        '''

        return 1.0 / (1.0 + np.exp(-1.0 * self._input))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient
        '''
        
        sigmoid_backward = self.output * (1.0 - self.output) 
        input_grad = sigmoid_backward * output_grad
        return input_grad
        
class Linear(Operation):
    '''
    Identity activation
    '''
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        '''
        Pass Through
        '''
        return self._input  
    
    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Pass Through
        '''
        return output_grad

class Tanh(Operation):
    '''
    Tanh Activation Function
    '''
    def __init__(self):
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        '''
        Compute output
        '''
        return np.tanh(self._input)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient
        '''
        
        return output_grad * (1 - self.output * self.output)