import math
import pandas as pd
import torch

class Config:
    def __init__(self, config_file: str, device: torch.device) -> None:
        self._device = device
        self._read(config_file)
        self._calc_params()

    def check(self) -> None:
        for k, v in vars(self).items():
            print(f"{k}: {v}")

    def _read(self, config_file: str) -> None:
        config_dict = pd.read_json(config_file, typ="series").to_dict()

        self._dimention                 = config_dict["dimention"]
        self._delta_time                = config_dict["delta_time"]
        self._num_timesteps             = config_dict["num_timesteps"]
        self._rest_density              = config_dict["rest_density"] 
        self._particle_mass             = config_dict["particle_mass"]
        self._num_neighbor_particles    = config_dict["num_neighbor_particles"]
        self._boundary_size             = torch.tensor(config_dict["boundary_size"], dtype=torch.int).to(self._device)
        self._fluid_size                = torch.tensor(config_dict["fluid_size"], dtype=torch.int).to(self._device)
        self._external_force            = torch.tensor(config_dict["external_force"], dtype=torch.float32).to(self._device)
        self._constraint_relaxation     = config_dict["constraint_relaxation"]
        self._allowable_compressibility = config_dict["allowable_compressibility"]
        self._num_max_iterations        = config_dict["num_max_iterations"]
        self._k                         = config_dict["k"] # artificial pressure 
        self._n                         = config_dict["n"] # artificial pressure 
        self._q                         = config_dict["q"] # artificial pressure 
        self._c                         = config_dict["c"] # XSPH Viscosity
        self._initial_velocity          = torch.tensor(config_dict["initial_velocity"], dtype=torch.float32).to(self._device)
        self._initial_position          = torch.tensor(config_dict["initial_position"], dtype=torch.float32).to(self._device)

    def _calc_params(self) -> None:
        self._num_boundary_particles = (torch.prod(self._boundary_size) - torch.prod(self._boundary_size - 2)).item()
        self._num_fluid_particles    = torch.prod(self._fluid_size).item()
        self._num_particles          = self._num_boundary_particles + self._num_fluid_particles
        self._volume                 = self._num_fluid_particles * self._particle_mass / self._rest_density
        self._effective_radius       = ((3.0 * self._num_neighbor_particles * self._volume) / (4.0 * self._num_fluid_particles * math.pi)) ** (1.0 / 3.0)
        self._particle_radius        = (math.pi / (6.0 * self._num_neighbor_particles)) ** (1.0 / 3.0) * self._effective_radius

    @property
    def dim(self) -> int:
        return self._dimention

    @property
    def dt(self) -> float:
        return self._delta_time

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def rho_0(self) -> float:
        return self._rest_density

    @property
    def m(self) -> float:
        return self._particle_mass

    @property
    def b_size(self) -> torch.Tensor:
        return self._boundary_size

    @property
    def f_size(self) -> torch.Tensor:
        return self._fluid_size
    
    @property
    def f_ext(self) -> torch.Tensor:
        return self._external_force

    @property
    def eps(self) -> float:
        return self._constraint_relaxation

    @property
    def eta(self) -> float:
        return self._allowable_compressibility

    @property
    def num_max_iterations(self) -> int:
        return self._num_max_iterations

    @property
    def k(self) -> float:
        return self._k

    @property
    def n(self) -> float:
        return self._n

    @property
    def q(self) -> float:
        return self._q

    @property
    def c(self) -> float:
        return self._c

    @property
    def initial_vel(self) -> torch.Tensor:
        return self._initial_velocity

    @property
    def initial_pos(self) -> torch.Tensor:
        return self._initial_position

    @property
    def num_boundary_particles(self) -> int:
        return self._num_boundary_particles

    @property
    def num_fluid_particles(self) -> int:
        return self._num_fluid_particles

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @ property
    def h(self) -> float:
        return self._effective_radius

    @property
    def r(self) -> float:
        return self._particle_radius