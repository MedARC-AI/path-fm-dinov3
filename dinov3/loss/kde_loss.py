import torch
from torch import nn
import torch.nn.functional as F


class KDELoss(nn.Module):
    def __init__(self, kappa: float = 5.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, student_output: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(student_output, p=2, dim=-1)
        similarity = normalized @ normalized.transpose(-2, -1)
        kernel = torch.exp(self.kappa * similarity)
        density = kernel.sum(dim=1)
        return -torch.log(density + 1e-9).mean()
