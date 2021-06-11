import numpy as np
from numpy import ndarray 

class Optimizer(object):
    '''
    Base class for a Neural Network Optimizer
    '''

    def __init__(self, lr: float = 0.01, final_lr: float = 0, decay_type: str = None):
        self. lr = lr
        self.final_lr = lr
        self.decay_type = decay_type
        self.first = True

    def step(self, epoch: int = 0 ) -> None:
        '''
        Every Optimzier must implement the step function for customised updates
        '''
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, param_grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        '''
        Update rule for Optimizer
        '''
        raise NotImplementedError()

    def _setup_decay(self) -> None:
        '''
        Setup Learning rate decay
        '''
        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1.0 / (self.max_epochs -1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1) 
    
    def _decay_lr(self) -> None:
        '''
        Decay learning rate
        '''
        if not self.decay_type:
            return
        if self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch
        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch    

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent Optimizer
    '''
    def __init__(self, lr: float = 0.01, final_lr: float = 0, decay_type: str = None):
        super().__init__(lr, final_lr, decay_type)
    

    def _update_rule(self, **kwargs) -> None:
        '''
        Update rule for SGD
        '''
        kwargs['param'] -= self.lr*kwargs['param_grad']

class SGDMomentum(Optimizer):
    '''
    Stochastic Gradient Descent with Momentum
    '''
    def __init__(self, lr: float, final_lr: float = 0, decay_type: str = None, momentum: float = 0.9):
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self) -> None:
        '''
        Custom Step rule to initialize velocities and perform step
        '''
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False
        
        for (param, param_grad, velocity) in zip(self.net.params(), self.net.param_grads(), self.velocities):
            self._update_rule(param=param, param_grad=param_grad, velocity=velocity)
    
    def _update_rule(self, **kwargs) -> None:
        '''
        Update rule for SGD with Momentum

        '''
        
        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.lr * kwargs['param_grad']

        kwargs['param'] -= kwargs['velocity']
