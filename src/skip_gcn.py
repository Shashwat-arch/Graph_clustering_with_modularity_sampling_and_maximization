import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.sparse import SparseTensor

class GCN(nn.Module):
    """PyTorch implementation of Graph Convolutional Network (GCN) layer with skip connections.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        activation (str or callable): Activation function ('selu' by default)
        skip_connection (bool): Whether to use skip connections
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'selu',
                 skip_connection: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_connection = skip_connection
        
        # Linear transformation
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        
        # Skip connection weight
        if skip_connection:
            self.skip_weight = nn.Parameter(torch.ones(out_channels))
        else:
            self.register_parameter('skip_weight', None)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Activation function
        if activation == 'selu':
            self.activation = F.selu
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.skip_weight is not None:
            nn.init.ones_(self.skip_weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: Tensor, norm_adjacency: SparseTensor) -> Tensor:
        """
        Args:
            features (Tensor): Node features [n_nodes, in_channels]
            norm_adjacency (SparseTensor): Normalized adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Tensor: Output node representations [n_nodes, out_channels]
        """
        # Linear transformation
        output = self.linear(features)
        
        # Graph convolution
        if self.skip_connection:
            output = output * self.skip_weight + torch.sparse.mm(norm_adjacency, output)
        else:
            output = torch.sparse.mm(norm_adjacency, output)
        
        # Add bias and apply activation
        output = output + self.bias
        return self.activation(output)