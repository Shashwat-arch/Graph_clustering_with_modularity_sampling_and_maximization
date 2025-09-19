import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, temperature=0.07, gamma=0.5, scale_by_temperature=True, alpha=0.01, coef0=0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma  # Previously RBF bandwidth, unused for sigmoid but kept for compatibility
        self.scale_by_temperature = scale_by_temperature
        self.alpha = alpha  # Sigmoid kernel alpha parameter
        self.coef0 = coef0  # Sigmoid kernel coef0 parameter


    def forward(self, out, mask):
        device = out.device
        
        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col = row.to(device), col.to(device)
        batch_size = out.shape[0]

        # Compute dot product between all pairs
        dot_product = torch.matmul(out, out.T)  # [batch, batch]

        # Compute Sigmoid kernel similarity: tanh(alpha * dot_product + coef0)
        sim_matrix = torch.tanh(self.alpha * dot_product + self.coef0)

        # Apply temperature scaling
        sim_matrix = torch.div(sim_matrix, self.temperature)

        # Numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        # Mask self-comparisons
        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size, device=device),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )

        # Compute probabilities with epsilon for numerical stability
        exp_logits = torch.exp(sim_matrix) * logits_mask
        sum_exp = exp_logits.sum(1, keepdim=True) + 1e-8  # Prevent log(0)
        log_probs = sim_matrix - torch.log(sum_exp)

        # Calculate loss using positive pairs
        log_probs_pos = log_probs[row, col]
        loss = -log_probs_pos.mean()

        if self.scale_by_temperature:
            loss *= self.temperature
            
        return loss
