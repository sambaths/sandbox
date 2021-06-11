from numpy import ndarray
import numpy as np

from utils import assert_same_shape, permute_data, softmax


class Loss(object):
    '''
    The loss of a neural Network.
    '''

    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
        Compute the loss
        '''

        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
        Compute the gradient of the loss with respect to the input to the loss function
        '''

        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
        Every subclass of Loss must implement _output function
        '''

        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
        Every subclass of Loss must implement _input_grad function
        '''

        raise NotImplementedError()


class MeanSquaredError(Loss):
    '''
    Mean Squared Error Loss
    '''
    def __init__(self):
        super().__init__()

    def _output(self) -> float:
        '''
        Compute the mean sqaured error loss
        '''
        
        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]
        return loss

    def _input_grad(self) -> ndarray:
        '''
        Compute the loss gradient with respect to the inputs to loss function
        '''

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]

class SoftmaxCrossEntropyLoss(Loss):
    '''
    Apply Softmax and Then Calculate CrossEntropy Loss
    '''
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.single_output = False

    def _output(self) -> float:
        '''
        Apply softmax and compute crossentropy loss
        '''

        softmax_preds = softmax(self.prediction, axis=1)
        
        # clipping for numerical stability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (1.0 - self.target) * np.log(1 - self.softmax_preds)

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        '''
        Compute the gradient with respect to the input to the loss function
        '''

        return self.softmax_preds - self.target
