from copy import deepcopy
import numpy as np
from numpy import ndarray

from typing import List, Tuple


from utils import assert_same_shape, permute_data, assert_dim

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

class Conv2D_Op_(ParamOperation):
    '''
    Implements the Convolution operation 
    '''
    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    
    def _pad_1d(self, inp: ndarray) -> ndarray:
        '''
        Pad a 1D array inorder to maintain its shape after the convolution
        '''
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self, inp: ndarray) ->ndarray:
        '''
        Pad a batch of 1D array
        '''
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self, inp: ndarray) -> ndarray:
        '''
        Pad a square 2D array
        '''
        inp_pad = self._pad_1d_batch(inp) # eg: inp -> 28, 28 : inp_pad -> 28, 30
        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2)) # 1, 30
        return np.concatenate([other, inp_pad, other]) # 30, 30

    def _pad_2d(self, inp: ndarray) -> ndarray:
        '''
        Pad a batch of 2D arrays
        '''
        outs = [self._pad_2d_obs(obs) for obs in inp]
        return np.stack(outs)

    def _pad_conv_input(self, inp: ndarray) -> ndarray:
        '''
        Padding the input
        '''

        return np.stack([self._pad_2d(obs) for obs in inp])

    def _compute_output_obs(self, obs: ndarray) -> ndarray:
        '''
        Compute the output of convolution function for a single observation
        '''

        assert_dim(obs, 3)
        assert_dim(self.param, 4)

        param_size = self.param.shape[2]
        # param_mid = param_size // 2
        obs_pad = self._pad_2d(obs)

        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]
        img_size = obs.shape[1]
        
        out = np.zeros((out_channels, ) + obs.shape[1:])
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_w in range(img_size):
                    for o_h in range(img_size):
                        for p_w in range(param_size):
                            for p_h in range(param_size):
                                out[c_out][o_w][o_h] += self.param[c_in][c_out][p_w][p_h] * obs_pad[c_in][o_w+p_w][o_h+p_h]

        return out
    
    def _output(self,
                inference: bool = False) -> ndarray:
        '''
        Output of applying convolution function for a batch of data
        '''
        
        outs = [self._compute_output_obs(obs) for obs in self._input]
        return np.stack(outs)

    def _compute_grads_obs(self, input_obs: ndarray, output_grad_obs: ndarray) -> ndarray:
        '''
        Computes the input gradient for a single observation
        '''
        input_grad = np.zeros_like(input_obs)
        param_size = self.param.shape[2]
        # param_mid = param_size // 2
        # img_size = input_obs.shape[1]
        in_channels = input_obs.shape[0]
        out_channels = self.param.shape[1]
        output_obs_pad = self._pad_2d(output_grad_obs)

        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for i_w in range(input_obs.shape[1]):
                    for i_h in range(input_obs.shape[2]):
                        for p_w in range(param_size):
                            for p_h in range(param_size):
                                input_grad[c_in][i_w][i_h] += output_obs_pad[c_out][i_w+param_size-p_w-1][i_h+param_size-p_h-1] * self.param[c_in][c_out][p_w][p_h]
        return input_grad

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Computes the Input Gradients for a batch of inputs
        '''

        grads = [self._compute_grads_obs(self._input[i], output_grad[i]) for i in range(output_grad.shape[0])]
        return np.stack(grads)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Computes the parameters gradients
        '''
        param_grad = np.zeros_like(self.param)
        param_size = self.param.shape[2]
        in_channels = self._input.shape[1]
        out_channels = output_grad.shape[1]

        inp_pad = self._pad_conv_input(self._input)
        img_shape = output_grad.shape[2:]

        for i in range(self._input.shape[0]):
            for c_in in range(in_channels):
                for c_out in range(out_channels):
                    for o_w in range(img_shape[0]):
                        for o_h in range(img_shape[1]):
                            for p_w in range(param_size):
                                for p_h in range(param_size):
                                    param_grad[c_in][c_out][p_w][p_h] += inp_pad[i][c_in][o_w+p_w][o_h+p_h] * output_grad[i][c_out][o_w][o_h]
        
        return param_grad


class Conv2D_Op(ParamOperation):
    
    def __init__(self, W: ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    
    def _pad_1d(self, inp: ndarray) -> ndarray:
        '''
        Pad a 1D array inorder to maintain its shape after the convolution
        '''
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self, inp: ndarray) ->ndarray:
        '''
        Pad a batch of 1D array
        '''
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self, inp: ndarray) -> ndarray:
        '''
        Pad a square 2D array
        '''
        inp_pad = self._pad_1d_batch(inp) # eg: inp -> 28, 28 : inp_pad -> 28, 30
        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2)) # 1, 30
        return np.concatenate([other, inp_pad, other]) # 30, 30

    def _pad_2d_channel(self, inp: ndarray):
        '''
        Pad a batch of 2D arrays
        '''

        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self, input_: ndarray):
        '''
        Get patches of the input image
        '''

        images_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in input_])
        patches = []
        img_height = images_batch_pad.shape[2]
        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch = images_batch_pad[:, :, h: h+self.param_size, w: w+self.param_size]
                patches.append(patch)

        return np.stack(patches)


    def _output(self, inference: bool = False):
        '''
        Applying the Convolution function to the batch using np.matmul 
        '''
        batch_size = self._input.shape[0]
        img_height = self._input.shape[2]
        img_size = self._input.shape[2] * self._input.shape[3]
        patch_size = self.param.shape[0] * self.param.shape[2] * self.param.shape[3]

        patches = self._get_image_patches(self._input)
               
        patches_reshaped = patches.transpose(1, 0, 2, 3, 4).reshape(batch_size, img_size, -1)

        param_reshaped = self.param.transpose(0, 2, 3, 1).reshape(patch_size, -1)
        output_reshaped = np.matmul(patches_reshaped, param_reshaped).reshape(batch_size, img_height, img_height, -1).transpose(0, 3, 1, 2)

        return output_reshaped

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Computes the Input Gradients for a batch of inputs using np.matmul
        '''
        batch_size = self._input.shape[0]
        img_size = self._input.shape[2] * self._input.shape[3]
        img_height = self._input.shape[2]

        output_patches = self._get_image_patches(output_grad).transpose(1, 0, 2, 3, 4).reshape(batch_size * img_size, -1)
        param_reshaped = self.param.reshape(self.param.shape[0], -1).transpose(1, 0)
        return np.matmul(output_patches, param_reshaped).reshape(batch_size, img_height, img_height, self.param.shape[0]).transpose(0, 3, 1, 2)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Computes the parameters gradients using np.matmul
        '''
        batch_size = self._input.shape[0]
        img_size = self._input.shape[2] * self._input.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = self._get_image_patches(self._input).reshape(batch_size * img_size, -1).transpose(1, 0)

        out_grad_reshape = output_grad.transpose(0, 2, 3, 1).reshape(batch_size * img_size, -1)

        return np.matmul(in_patches_reshape, out_grad_reshape).reshape(in_channels, self.param_size, self.param_size, out_channels).transpose(0, 3, 1, 2)

