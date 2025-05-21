import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.sparse import SparseTensor

class DMoN(nn.Module):
    """PyTorch implementation of Deep Modularity Network (DMoN) layer.
    
    Args:
        n_clusters (int): Number of clusters
        collapse_regularization (float): Collapse regularization weight
        dropout_rate (float): Dropout rate before softmax
        do_unpooling (bool): Whether to perform unpooling
    """
    
    def __init__(self, 
                 n_clusters: int,
                 collapse_regularization: float = 0.1,
                 dropout_rate: float = 0,
                 do_unpooling: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling

        # Transformation layers
        self.transform = nn.Sequential(
            nn.Linear(in_features=0, out_features=n_clusters),  # Placeholder
            nn.Dropout(dropout_rate))
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization for linear layer
        nn.init.orthogonal_(self.transform[0].weight)
        nn.init.zeros_(self.transform[0].bias)

    def forward(self, features: Tensor, adjacency: SparseTensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features (Tensor): Node features [n_nodes, n_features]
            adjacency (SparseTensor): Sparse adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Tuple[Tensor, Tensor]: (pooled_features, assignments)
        """
        # Dynamic input dimension handling
        if self.transform[0].in_features == 0:
            self.transform[0].in_features = features.size(1)
            self.transform[0].reset_parameters()

        # Cluster assignments
        assignments = F.softmax(self.transform(features), dim=1)  # [n, k]
        
        # Cluster sizes and normalized assignments
        cluster_sizes = assignments.sum(dim=0)  # [k]
        assignments_pooling = assignments / cluster_sizes  # [n, k]

        # Degree calculations
        degrees = torch.sparse.sum(adjacency, dim=1).to_dense().unsqueeze(1)  # [n, 1]
        n_nodes = adjacency.size(0)
        n_edges = degrees.sum() / 2

        # Graph pooling
        adj_assign = torch.sparse.mm(adjacency, assignments)  # [n, k]
        graph_pooled = assignments.t() @ adj_assign  # [k, k]

        # Normalizer calculation
        normalizer_left = assignments.t() @ degrees  # [k, 1]
        normalizer_right = degrees.t() @ assignments  # [1, k]
        normalizer = (normalizer_left @ normalizer_right) / (2 * n_edges)

        # Loss calculations
        spectral_loss = -torch.trace(graph_pooled - normalizer) / (2 * n_edges)
        collapse_loss = (torch.norm(cluster_sizes) / n_nodes * 
                        torch.sqrt(torch.tensor(self.n_clusters)) - 1)

        # Store losses as attributes
        self.spectral_loss = spectral_loss
        self.collapse_loss = self.collapse_regularization * collapse_loss

        # Feature pooling
        features_pooled = assignments_pooling.t() @ features  # [k, d]
        features_pooled = F.selu(features_pooled)

        if self.do_unpooling:
            features_pooled = assignments_pooling @ features_pooled  # [n, d]

        return features_pooled, assignments

    def get_losses(self) -> Tuple[Tensor, Tensor]:
        """Returns the spectral loss and collapse loss"""
        return self.spectral_loss, self.collapse_loss