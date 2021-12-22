import time
import pandas as pd
import torch
from torch_geometric.nn.pool import radius_graph
from config import Config
from solid import Box
from particle import Particle
from calc import calc_density, calc_position_correction, calc_xsph_viscosity, calc_divergence

class PositionBasedFluids:
    def __init__(self, device: torch.device) -> None:
        self._device = device

    def reset(self, config_file: str) -> None:
        self._config = Config(config_file, self._device)
        self._timestep = 0
        self._solid = Box(self._config.b_size, self._config.r)

        self._particle = Particle(self._config, self._device) 

        # to measure performance
        self._elapsed_time_s  = []
        self._compressibility = []
        self._divergence_vel  = []

    def run(self, output_file: str, measurement_file: str) -> None:
        while self._timestep < self._config.num_timesteps:
            start_time_s = time.time()

            self._evolve()

            end_time_s = time.time()
            elapsed_time_s = end_time_s - start_time_s

            self._write(output_file)

            self._measure(elapsed_time_s)

            self._timestep += 1

        self._write_measurement(measurement_file)

    def check(self) -> None:
        for k, v in vars(self).items():
            print(f"{k}: {v}")

    def _evolve(self) -> None:
        # update particles
        self._particle.f_vel = self._particle.f_next_vel.clone()
        self._particle.f_pos = self._particle.f_next_pos.clone()

        # apply external force
        self._particle.f_next_vel += self._config.f_ext * self._config.dt
        self._particle.f_next_pos += self._particle.f_next_vel * self._config.dt

        # process boundary 
        self._particle.f_next_pos = self._solid.respond(self._particle.f_next_pos)

        # intermidiate quantities
        self._particle.mid_vel = self._particle.next_vel.clone()
        self._particle.mid_pos = self._particle.next_pos.clone()

        # position correction
        edge_index = radius_graph(self._particle.next_pos, self._config.h, loop=True)
        itr = 0
        cmp_var = 1.0
        while itr < self._config.num_max_iterations and cmp_var > self._config.eta:
            dp = calc_position_correction(self._config.rho_0, self._config.h, self._config.m, self._config.dt, self._config.q, self._config.k, self._config.n, self._config.eps, self._particle.vol, self._particle.next_pos, edge_index)
            self._particle.f_next_pos += dp[self._config.num_boundary_particles:]
            itr += 1
            rho = calc_density(self._config.rho_0, self._config.h, self._particle.vol, self._particle.next_pos, edge_index)
            cmp_var = torch.mean(torch.abs(rho / self._config._rest_density - 1.0))

        # delta position
        self._particle.dp = self._particle.next_pos - self._particle.pos

        # process boundary
        self._particle.f_next_pos = self._solid.respond(self._particle.f_next_pos)

        # update velocity
        self._particle.f_next_vel = (self._particle.f_next_pos - self._particle.f_pos) / self._config.dt

        # XSPH viscosity
        xsph_viscosity = calc_xsph_viscosity(self._config.rho_0, self._config.h, self._config.c, self._particle.vol, self._particle.next_vel, self._particle.next_pos, edge_index)
        self._particle.f_next_vel += xsph_viscosity[self._config.num_boundary_particles:]

    def _measure(self, elapsed_time_s: float) -> None:
        edge_index = radius_graph(self._particle.next_pos, self._config.h, loop=True)

        # elapsed time
        self._elapsed_time_s.append(elapsed_time_s)

        # compressibility
        rho = calc_density(self._config.rho_0, self._config.h, self._particle.vol, self._particle.next_pos, edge_index)
        cmp = torch.mean(torch.abs(rho / self._config.rho_0 - 1.0)).item()
        self._compressibility.append(cmp)

        # divergence
        div = torch.mean(torch.abs(calc_divergence(self._config.rho_0, self._config.h, self._particle.vol, self._particle.next_pos, self._particle.next_vel, edge_index))).item()
        self._divergence_vel.append(div)

    def _write(self, output_file: str) -> None:
        df = pd.DataFrame({
            "timestep": self._timestep,

            "lab": self._particle.lab.cpu(),

            "vol": self._particle.vol.cpu(),

            "vel.x": self._particle.vel[:, 0].cpu(),
            "vel.y": self._particle.vel[:, 1].cpu(),
            "vel.z": self._particle.vel[:, 2].cpu(),

            "pos.x": self._particle.pos[:, 0].cpu(),
            "pos.y": self._particle.pos[:, 1].cpu(),
            "pos.z": self._particle.pos[:, 2].cpu(),

            "mid_vel.x": self._particle.mid_vel[:, 0].cpu(),
            "mid_vel.y": self._particle.mid_vel[:, 1].cpu(),
            "mid_vel.z": self._particle.mid_vel[:, 2].cpu(),

            "mid_pos.x": self._particle.mid_pos[:, 0].cpu(),
            "mid_pos.y": self._particle.mid_pos[:, 1].cpu(),
            "mid_pos.z": self._particle.mid_pos[:, 2].cpu(),

            "dp.x": self._particle.dp[:, 0].cpu(),
            "dp.y": self._particle.dp[:, 1].cpu(),
            "dp.z": self._particle.dp[:, 2].cpu(),
        })

        header = True if self._timestep == 0 else False
        df.to_csv(output_file, index=False, header=header, mode="a")

    def _write_measurement(self, measurement_file: str) -> None:
        df = pd.DataFrame({
            "elapsed_time_s": self._elapsed_time_s,
            "compressibility": self._compressibility,
            "divergence_vel": self._divergence_vel
        })
        df.to_csv(measurement_file, index=False, mode="w")