import torch
from torch_geometric.nn.pool import radius_graph
from torch_scatter import scatter_add
from config import Config
from kernel import poly6

class Particle:
    def __init__(self, config: Config, device: torch.device) -> None:
        self._num_boundary_particles = config.num_boundary_particles
        self._device = device

        self._allocate(config)
        self._reset(config)

    def _allocate(self, config: Config) -> None:
        self._lab      = torch.empty(config.num_particles, dtype=torch.long).to(self._device)
        self._vol      = torch.empty(config.num_particles, dtype=torch.float32).to(self._device)
        self._vel      = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)
        self._pos      = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)
        self._mid_vel  = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)
        self._mid_pos  = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)
        self._next_vel = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)
        self._next_pos = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)
        self._dp       = torch.empty((config.num_particles, config.dim), dtype=torch.float32).to(self._device)

    def _reset(self, config: Config) -> None:
        # label
        self.b_lab = 0
        self.f_lab = 1

        # velocity
        self.b_vel = torch.zeros(config.dim, dtype=torch.float32).to(self._device)
        self.f_vel = config.initial_vel.clone()

        # position
        b_min = -(config.b_size - 1) * config.r
        idx = 0
        for x in range(config.b_size[0]):
            for y in range(config.b_size[1]):
                for z in range(config.b_size[2]):
                    if (1 <= x < config.b_size[0]-1) and (1 <= y < config.b_size[1]-1) and (1 <= z < config.b_size[2]-1):
                        continue
                    self.b_pos[idx] = b_min + 2.0 * config.r * torch.tensor([x, y, z]).to(self._device)
                    idx += 1

        f_min = config.initial_pos - (config.f_size - 1) * config.r
        idx = 0
        for x in range(config.f_size[0]):
            for y in range(config.f_size[1]):
                for z in range(config.f_size[2]):
                    self.f_pos[idx] = f_min + 2.0 * config.r * torch.tensor([x, y, z]).to(self._device)
                    idx += 1

        # volume
        edge_index = radius_graph(self.b_pos, config.h, loop=True)
        i = edge_index[1]
        j = edge_index[0]
        r = torch.linalg.norm(self.b_pos[i] - self.b_pos[j], axis=1)
        rho = scatter_add(config.m * poly6(r, config.h), index=i, dim_size=config.num_boundary_particles)
        self.b_vol = config.m / rho

        self.f_vol = config.m / config.rho_0

        # next 
        self.next_vel = self.vel.clone()
        self.next_pos = self.pos.clone()

    @property
    def lab(self) -> torch.Tensor:
        return self._lab[:]

    @lab.setter
    def lab(self, value: torch.Tensor) -> None:
        self._lab[:] = value

    @property
    def b_lab(self) -> torch.Tensor:
        return self._lab[:self._num_boundary_particles]

    @b_lab.setter
    def b_lab(self, value: torch.Tensor) -> None:
        self._lab[:self._num_boundary_particles] = value

    @property
    def f_lab(self) -> torch.Tensor:
        return self._lab[self._num_boundary_particles:]

    @f_lab.setter
    def f_lab(self, value: torch.Tensor) -> None:
        self._lab[self._num_boundary_particles:] = value

    @property
    def vol(self) -> torch.Tensor:
        return self._vol[:]

    @vol.setter
    def vol(self, value: torch.Tensor) -> None:
        self._vol[:] = value

    @property
    def b_vol(self) -> torch.Tensor:
        return self._vol[:self._num_boundary_particles]

    @b_vol.setter
    def b_vol(self, value: torch.Tensor) -> None:
        self._vol[:self._num_boundary_particles] = value

    @property
    def f_vol(self) -> torch.Tensor:
        return self._vol[self._num_boundary_particles:]

    @f_vol.setter
    def f_vol(self, value: torch.Tensor) -> None:
        self._vol[self._num_boundary_particles:] = value

    @property
    def vel(self) -> torch.Tensor:
        return self._vel[:]

    @vel.setter
    def vel(self, value: torch.Tensor) -> None:
        self._vel[:] = value

    @property
    def b_vel(self) -> torch.Tensor:
        return self._vel[:self._num_boundary_particles]

    @b_vel.setter
    def b_vel(self, value: torch.Tensor) -> None:
        self._vel[:self._num_boundary_particles] = value

    @property
    def f_vel(self) -> torch.Tensor:
        return self._vel[self._num_boundary_particles:]

    @f_vel.setter
    def f_vel(self, value: torch.Tensor) -> None:
        self._vel[self._num_boundary_particles:] = value

    @property
    def pos(self) -> torch.Tensor:
        return self._pos[:]

    @pos.setter
    def pos(self, value: torch.Tensor) -> None:
        self._pos[:] = value

    @property
    def b_pos(self) -> torch.Tensor:
        return self._pos[:self._num_boundary_particles]

    @b_pos.setter
    def b_pos(self, value: torch.Tensor) -> None:
        self._pos[:self._num_boundary_particles] = value

    @property
    def f_pos(self) -> torch.Tensor:
        return self._pos[self._num_boundary_particles:]

    @f_pos.setter
    def f_pos(self, value: torch.Tensor) -> None:
        self._pos[self._num_boundary_particles:] = value

    @property
    def mid_vel(self) -> torch.Tensor:
        return self._mid_vel[:]

    @mid_vel.setter
    def mid_vel(self, value: torch.Tensor) -> None:
        self._mid_vel[:] = value

    @property
    def b_mid_vel(self) -> torch.Tensor:
        return self._mid_vel[:self._num_boundary_particles]

    @b_mid_vel.setter
    def b_mid_vel(self, value: torch.Tensor) -> None:
        self._mid_vel[:self._num_boundary_particles] = value

    @property
    def f_mid_vel(self) -> torch.Tensor:
        return self._mid_vel[self._num_boundary_particles:]

    @f_mid_vel.setter
    def f_mid_vel(self, value: torch.Tensor) -> None:
        self._mid_vel[self._num_boundary_particles:] = value

    @property
    def mid_pos(self) -> torch.Tensor:
        return self._mid_pos[:]

    @mid_pos.setter
    def mid_pos(self, value: torch.Tensor) -> None:
        self._mid_pos[:] = value

    @property
    def b_mid_pos(self) -> torch.Tensor:
        return self._mid_pos[:self._num_boundary_particles]

    @b_mid_pos.setter
    def b_mid_pos(self, value: torch.Tensor) -> None:
        self._mid_pos[:self._num_boundary_particles] = value

    @property
    def f_mid_pos(self) -> torch.Tensor:
        return self._mid_pos[self._num_boundary_particles:]

    @f_mid_pos.setter
    def f_mid_pos(self, value: torch.Tensor) -> None:
        self._mid_pos[self._num_boundary_particles:] = value

    @property
    def next_vel(self) -> torch.Tensor:
        return self._next_vel[:]

    @next_vel.setter
    def next_vel(self, value: torch.Tensor) -> None:
        self._next_vel[:] = value

    @property
    def b_next_vel(self) -> torch.Tensor:
        return self._next_vel[:self._num_boundary_particles]

    @b_next_vel.setter
    def b_next_vel(self, value: torch.Tensor) -> None:
        self._next_vel[:self._num_boundary_particles] = value

    @property
    def f_next_vel(self) -> torch.Tensor:
        return self._next_vel[self._num_boundary_particles:]

    @f_next_vel.setter
    def f_next_vel(self, value: torch.Tensor) -> None:
        self._next_vel[self._num_boundary_particles:] = value

    @property
    def next_pos(self) -> torch.Tensor:
        return self._next_pos[:]

    @next_pos.setter
    def next_pos(self, value: torch.Tensor) -> None:
        self._next_pos[:] = value

    @property
    def b_next_pos(self) -> torch.Tensor:
        return self._next_pos[:self._num_boundary_particles]

    @b_next_pos.setter
    def b_next_pos(self, value: torch.Tensor) -> None:
        self._next_pos[:self._num_boundary_particles] = value

    @property
    def f_next_pos(self) -> torch.Tensor:
        return self._next_pos[self._num_boundary_particles:]

    @f_next_pos.setter
    def f_next_pos(self, value: torch.Tensor) -> None:
        self._next_pos[self._num_boundary_particles:] = value

    @property
    def dp(self) -> torch.Tensor:
        return self._dp[:]

    @dp.setter
    def dp(self, value: torch.Tensor) -> None:
        self._dp[:] = value

    @property
    def b_dp(self) -> torch.Tensor:
        return self._dp[:self._num_boundary_particles]

    @b_dp.setter
    def b_dp(self, value: torch.Tensor) -> None:
        self._dp[:self._num_boundary_particles] = value

    @property
    def f_dp(self) -> torch.Tensor:
        return self._dp[self._num_boundary_particles:]

    @f_dp.setter
    def f_dp(self, value: torch.Tensor) -> None:
        self._dp[self._num_boundary_particles:] = value