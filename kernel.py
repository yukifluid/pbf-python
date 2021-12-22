import math
import torch

def poly6(r: torch.Tensor, h: float) -> torch.Tensor:
    is_valid = torch.logical_and(0.0 <= r , r <= h)
    a = 315 / (64*math.pi*h**9)
    q = h**2 - r**2
    y = torch.zeros_like(r)
    y[is_valid] = a * q[is_valid]**3
    return y

def grad_spiky(r, r_ij, h):
    is_valid = torch.logical_and(0.0 < r , r <= h)
    a = -45 / (math.pi*h**6.0)
    q = (h-r).view(-1, 1)
    y = torch.zeros_like(r_ij)
    y[is_valid] = a * q[is_valid]**2.0 * r_ij[is_valid] / r.view(-1, 1)[is_valid]
    return y