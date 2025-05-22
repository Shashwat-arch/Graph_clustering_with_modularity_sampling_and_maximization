import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

class DMoNClustering(nn.Module):
    """Neural Network for Clustering Embeddings using DMoN-inspired approach.
    
    Args:
        input_dim (int): Dimension of input embeddings (512 in your case)
        n_clusters (int): Number of clusters to predict
        hidden_dim (int): Hidden layer dimension (default: 256)
        dropout (float): Dropout rate (default: 0.2)
        collapse_reg (float): Collapse regularization weight (default: 1.0)
    """
    
    def __init__(self, input_dim=512, n_clusters=10, hidden_dim=256, 
                 dropout=0.2, collapse_reg=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.collapse_reg = collapse_reg
        
        # Cluster assignment network
        self.assign_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_clusters),
            nn.Softmax(dim=1)  # Softmax over clusters
        )
        
    def forward(self, embeddings: Tensor, adjacency: SparseTensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            embeddings: Input embeddings [num_nodes, input_dim] (torch.float32)
            adjacency: Sparse adjacency matrix [num_nodes, num_nodes] (torch.sparse_coo)
            
        Returns:
            Tuple[Tensor, Tensor]: (cluster_assignments, pooled_embeddings)
                - cluster_assignments: [num_nodes, n_clusters]
                - pooled_embeddings: [n_clusters, input_dim] if not unpooling
        """
        # Get cluster assignments
        assignments = self.assign_net(embeddings)  # [N, K]
        # print("assignment: ", assignments)
        
        # Calculate losses
        spectral_loss, collapse_loss = self._calculate_losses(assignments, adjacency)
        # print("spectral loss: ", spectral_loss, "collapse loss: ", collapse_loss)
        
        # Pool embeddings by cluster
        cluster_sizes = assignments.sum(dim=0)  # [K]
        assignments_pooling = assignments / (cluster_sizes + 1e-8)  # [N, K]
        pooled_embeddings = assignments_pooling.T @ embeddings  # [K, D]
        
        return assignments, pooled_embeddings, spectral_loss, collapse_loss
    
    def _calculate_losses(self, assignments: Tensor, adjacency: SparseTensor) -> tuple[Tensor, Tensor]:
        """Compute DMoN losses"""
        # Degrees and edge count
        degrees = adjacency.sum(dim=1).to_dense()  # Correct for SparseTensor
        m = degrees.sum() / 2
        # print("Degrees and m calculated: ", degrees, m)
        
        # Spectral loss (modularity)
        adjacency = adjacency.to_dense()
        graph_pooled = assignments.T @ torch.sparse.mm(adjacency, assignments)  # [K, K]
        normalizer = (assignments.T @ degrees) @ (degrees @ assignments) / (2 * m)
        spectral_loss = -torch.trace(graph_pooled - normalizer) / (2 * m)
        
        # Collapse regularization
        cluster_sizes = assignments.sum(dim=0)  # [K]
        collapse_loss = (torch.norm(cluster_sizes) / adjacency.size(0) * \
                       torch.sqrt(torch.tensor(self.n_clusters)) - 1)

        # Entropy regularization (encourage balanced cluster usage)
        entropy = -torch.sum(assignments * torch.log(assignments + 1e-8)) / assignments.size(0)
        entropy_loss = -0.1 * entropy  # adjust the weight as needed
        
        # Return all losses
        total_loss = spectral_loss + self.collapse_reg * collapse_loss + entropy_loss
        return spectral_loss, self.collapse_reg * collapse_loss, entropy_loss