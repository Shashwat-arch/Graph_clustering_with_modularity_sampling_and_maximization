import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

class DMoNLoss(nn.Module):
    """DMoN loss calculation (PyTorch version)"""
    def __init__(self, n_clusters, temperature=0.07, scale_by_temperature=True, scale_by_weight=False, collapse_reg=1.0):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight
        self.n_clusters = n_clusters
        self.collapse_reg = collapse_reg

    def forward(self, out, mask, assignment, adjacency):

        ##--------------------------dot-product---------------------------------

        device = (torch.device('cuda') if out.is_cuda else torch.device('cpu'))

        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # compute logits
        dot = torch.matmul(out, out.T)
        dot = torch.div(dot, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(dot) * logits_mask
        log_probs = dot - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        labels = row.view(row.shape[0], 1)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        log_probs = log_probs[row, col]

        log_probs = log_probs.view(-1, 1)
        loss = torch.zeros_like(unique_labels, dtype=torch.float).to(device)
        loss.scatter_add_(0, labels, log_probs)
        loss = -1 * loss / labels_count.float().unsqueeze(1)

        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()

        ##----------------------------------------------------------------------

        ##--------------------------DMoN----------------------------------------
       # Suppose assignment is of shape [N], with integer cluster IDs
        assignment_one_hot = F.one_hot(assignment, num_classes=self.n_clusters).float()  # [N, K]

        # Calculate degrees
        degrees = adjacency.sum(dim=1)  # [N]
        m = degrees.sum() / 2.0  # scalar

        # Calculate spectral loss
        cluster_sizes = assignment_one_hot.sum(dim=0)  # [K]
        normalizer = torch.matmul(
            torch.matmul(assignment_one_hot.T, degrees.unsqueeze(-1)),  # [K, 1]
            torch.matmul(degrees.unsqueeze(0), assignment_one_hot)      # [1, K]
        ) / (2 * m)

        graph_pooled = torch.matmul(
            assignment_one_hot.T, torch.spmm(adjacency, assignment_one_hot)
        )  # [K, K]

        spectral_loss = -torch.trace(graph_pooled - normalizer) / (2 * m)

        # Calculate collapse regularization
        collapse_loss = (torch.norm(cluster_sizes) / adjacency.shape[0]) * \
                        torch.sqrt(torch.tensor(self.n_clusters, dtype=torch.float32, device=cluster_sizes.device)) - 1

        ##----------------------------------------------------------------------
        
        # Combine losses
        return loss + (spectral_loss + self.collapse_reg * collapse_loss)