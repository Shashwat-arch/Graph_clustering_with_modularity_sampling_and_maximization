import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from loss_functions.dmonloss import DMoNLoss

class GCNWithSkip(nn.Module):
    """Graph Convolutional Network with skip connection (PyTorch version)"""
    def __init__(self, input_dim, output_dim, activation=nn.SELU()):
        super().__init__()
        self.activation = activation
        self.kernel = nn.Linear(input_dim, output_dim)
        self.skip_weight = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, x, adjacency):
        # Transform features
        transformed = self.kernel(x)
        
        # Sparse matrix multiplication
        propagated = torch.spmm(adjacency, transformed)
        
        # Skip connection
        output = self.skip_weight * transformed + propagated
        return self.activation(output)

class DMoNModel(nn.Module):
    """Integrated DMoN Clustering Model"""
    def __init__(self, encoder, n_clusters, hidden_dim=64, collapse_reg=1.0):
        super().__init__()
        self.encoder = encoder
        self.gcn1 = GCNWithSkip(encoder.hidden_channels[-1], hidden_dim)
        self.gcn2 = GCNWithSkip(hidden_dim, hidden_dim)
        self.assign_layer = nn.Linear(hidden_dim, n_clusters)
        self.dmon_loss = DMoNLoss(n_clusters, collapse_reg)
        
    def forward(self, x, edge_index, adjacency):
        # Get initial embeddings from existing encoder
        x = self.encoder(x, edge_index)
        
        # DMoN specific processing
        x = self.gcn1(x, adjacency)
        x = self.gcn2(x, adjacency)
        assignments = F.softmax(self.assign_layer(x), dim=-1)
        return assignments

    def loss(self, assignments, adjacency):
        return self.dmon_loss(assignments, adjacency)