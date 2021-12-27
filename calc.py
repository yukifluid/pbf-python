import torch
from torch_scatter import scatter_add

from kernel import grad_spiky, poly6

def calc_divergence(rho_0: float, h: float, vol: torch.Tensor, p: torch.Tensor, phi: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    i = edge_index[1]
    j = edge_index[0]

    m = rho_0 * vol
    rho = calc_density(rho_0, h, vol, p, edge_index)
    r_ij = p[i] - p[j]
    r = torch.linalg.norm(r_ij, axis=1)
    div = scatter_add(m[j].view(-1, 1) * phi[j] / rho[j].view(-1, 1) * grad_spiky(r, r_ij, h), index=i, dim=0, dim_size=len(p))
    return div

def calc_density(rho_0: float, h: float, vol: torch.Tensor, p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    i = edge_index[1]
    j = edge_index[0]

    r = torch.linalg.norm(p[i] - p[j], axis=1)
    m = rho_0 * vol
    rho = scatter_add(m[j] * poly6(r, h), index=i, dim_size=len(p))
    return rho

def calc_position_correction(rho_0: float, h: float, m: float, dt: float, q: float, k: float, n: float, eps: float, vol: torch.Tensor, p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    i = edge_index[1]
    j = edge_index[0] 

    dq = torch.tensor(q * h, dtype=torch.float32)
    wq = poly6(dq, h)
    r_ij = p[i] - p[j]
    r = torch.linalg.norm(r_ij, axis=1)
    ww_j = poly6(r, h) / wq
    corr_j = -k * ww_j ** n * dt ** 2.0
    factor = calc_scaling_factor(rho_0, h, m, eps, vol, p, edge_index)
    dp_j = (factor[i] + factor[j] + corr_j).unsqueeze(dim=1) * grad_spiky(r, r_ij, h) / rho_0
    dp = scatter_add(dp_j, index=i, dim=0, dim_size=len(p))    
    return dp

def calc_scaling_factor(rho_0: float, h: float, m: float, eps: float, vol: torch.Tensor, p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    i = edge_index[1]
    j = edge_index[0]

    rho = calc_density(rho_0, h, vol, p, edge_index)
    c = rho / rho_0 - 1.0
    r_ij = p[i] - p[j]
    r = torch.linalg.norm(r_ij, axis=1)
    relative_m = (rho_0 * vol / m).unsqueeze(dim=1)
    dp_j = -relative_m[j] * grad_spiky(r, r_ij, h) / rho_0
    sd = scatter_add(torch.sum(dp_j**2.0, axis=1), index=i, dim_size=len(p)) + torch.sum(scatter_add(-dp_j, index=i, dim=0, dim_size=len(p))**2.0, axis=1)
    fact = -c / (sd + eps)
    return fact

def calc_xsph_viscosity(rho_0: float, h: float, c: float, vol: torch.Tensor, rho: torch.Tensor, v: torch.Tensor, p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    i = edge_index[1]
    j = edge_index[0]

    v_ji = v[j] - v[i]
    r_ji = p[j] - p[i]
    r = torch.linalg.norm(r_ji, axis=1)
    m = rho_0 * vol
    xsph_viscosity = c * scatter_add(torch.unsqueeze(m[j] / rho[j] * poly6(r, h), dim=1) * v_ji, index=i, dim=0, dim_size=len(p))
    return xsph_viscosity