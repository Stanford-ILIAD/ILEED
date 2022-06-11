'''
This file contains any NN models we may need in our experiments
'''
#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from torch import Tensor, as_tensor, no_grad, manual_seed
from torch.nn import Module, Sequential, Linear, ReLU, Conv2d, Flatten
from torch.nn.functional import softmax, relu

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box

class MLP(Module):
    '''
    MLP class with three linear layers with two ReLU non-linearities between,
    uses softmax at the output if called via prob_forward() method.
    '''
    def __init__(self, input_size, hidden_size, output_size, device, seed=123):
        super(MLP, self).__init__()
        self.seed = manual_seed(seed)
        self.device = device
        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(inplace=True),
            Linear(hidden_size, hidden_size),
            ReLU(inplace=True),
            Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)
    
    def prob_forward(self, x):
        return softmax(self.net(x), dim=-1)

