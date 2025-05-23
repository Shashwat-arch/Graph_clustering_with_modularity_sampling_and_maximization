import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.cluster import KMeans
from typing import Optional

class DEC_Clustering(nn.Module):
    """Deep Embedded Clustering adapted for graph embeddings with spectral initialization"""
    
    def __init__(self, input_dim=512, n_clusters=10, hidden_dim=256, 
                 alpha=0.8, dropout=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_dim = input_dim
        
        # Non-linear projection network
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, input_dim)  # Added to match original dim
        )
        
        # Cluster assignment layer
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        
    def forward(self, embeddings: Tensor, adjacency: Optional[Tensor] = None) -> tuple:
        """
        Args:
            embeddings: Input embeddings [num_nodes, input_dim]
            adjacency: Optional (unused here but kept for compatibility)
        Returns:
            tuple: (assignments, pooled_embeddings, kl_loss, reconstruction_loss, total_loss)
        """
        # Project embeddings
        z_hidden = self.projection[:-1](embeddings)  # [N, hidden_dim] for clustering
        z_recon = self.projection(embeddings)  # [N, input_dim] for reconstruction
        
        # Student's t-distribution for soft assignment
        q = 1.0 / (1.0 + (torch.cdist(z_hidden, self.cluster_centers)**2 / self.alpha))
        q = q**((self.alpha + 1.0) / 2.0)
        assignments = (q.t() / torch.sum(q, dim=1)).t()  # [N, K]
        
        # Compute target distribution
        p = target_distribution(assignments.detach())
        
        # Loss calculations
        kl_loss = F.kl_div(assignments.log(), p, reduction='batchmean')
        recon_loss = F.mse_loss(z_recon, embeddings)  # Now same dimensions
        collapse_loss = self._calculate_collapse_loss(assignments)
        
        total_loss = kl_loss + 0.1 * recon_loss + 0.01 * collapse_loss
        
        # Pool embeddings
        pooled = torch.mm(assignments.t(), z_hidden)  # [K, hidden_dim]
        
        return assignments, pooled, kl_loss, recon_loss, total_loss, z_hidden
    
    def _calculate_collapse_loss(self, assignments: Tensor) -> Tensor:
        cluster_sizes = assignments.sum(0)
        return torch.std(cluster_sizes) / torch.mean(cluster_sizes)
    
    def initialize_clusters(self, embeddings: Tensor):
        """Initialize cluster centers using k-means on projected embeddings"""
        with torch.no_grad():
            z = self.projection[:-1](embeddings)  # Use hidden representation
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            kmeans.fit(z.cpu().numpy())
            self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, 
                                                   dtype=torch.float32, 
                                                   device=embeddings.device)

def target_distribution(q: Tensor) -> Tensor:
    """Compute target distribution for KL divergence loss"""
    weight = (q**2) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()
