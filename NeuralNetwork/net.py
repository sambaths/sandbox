from copy import deepcopy
from numpy import ndarray
import numpy as np

from typing import List, Tuple

from utils import assert_same_shape, permute_data
from layers import Layer
from loss import Loss
from optim import Optimizer

class NeuralNetwork(object):
    '''
    class for a Neural Network
    '''
    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, 'seed', self.seed)

    def forward(self, x_batch: ndarray, inference: bool = False) -> ndarray:
        '''
        Passes data forward through a series of layers
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
        Passes data backward through a series of layers
        '''
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray, inference: bool = False) -> float:
        '''
        Passes data forward through the layers
        '''

        predictions = self.forward(x_batch, inference)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())

        return loss

    def params(self):
        '''
        Get the parameters of the network
        '''

        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        Get the gradient of the loss with respect to the parameters for the network
        '''
        
        for layer in self.layers:
            yield from layer.param_grads


class Trainer(object):
    '''
    Trains a neural Network
    '''
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        setattr(self.optim, 'net', self.net)

    def fit(self, X_train: ndarray, y_train: ndarray, 
                  X_test: ndarray, y_test: ndarray,
                  epochs: int = 100,
                  eval_every: int = 10,
                  batch_size: int = 32,
                  seed: int = 1,
                  restart: bool = True) -> None:
        '''
        Fits the neural network on the training data for a certain number of epochs.
        Every 'eval_every' epochs, evaluates the network on testing data
        '''
        setattr(self.optim, 'max_epochs', epochs)
        self.optim._setup_decay()

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9
        for epoch in range(epochs):
            if (epoch + 1) % eval_every == 0:
                # for early stopping 
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for i, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (epoch + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)
                
                if loss < self.best_loss:
                    print(f'Validation loss after {epoch + 1} epochs is {loss:.3f}')
                    self.best_loss = loss
                else:
                    print(f'Loss increased after {epoch + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {epoch + 1 - eval_every}')
                    self.net = last_model
                    setattr(self.optim, 'net', self.net)
                    break
            if self.optim.final_lr:
                self.optim._decay_lr()

    def generate_batches(self, X: ndarray, y: ndarray, batch_size: int = 32) -> Tuple[ndarray]:
        '''
        Generates batches for training
        '''
        assert X.shape[0] == y.shape[0], f'''Features and targets must have the same number of rows, 
                                instead features has {X.shape[0]} and target has {y.shape[0]}'''

        N = X.shape[0]
        for i in range(0, N, batch_size):
            X_batch, y_batch = X[i: i + batch_size], y[i: i + batch_size]
            yield X_batch, y_batch
