import sys
import json
import torch

if len(sys.argv) < 2:
    print("not enough arguments", file=sys.stderr)
    exit(1)

scene_path = sys.argv[1]
config_file = f"{scene_path}/config.json"
delta_time = 0.016
particle_radius = 0.017109517455886

position_min = -(torch.tensor([10, 10, 10]) - 1) * particle_radius
position_max =  (torch.tensor([10, 10, 10]) - 1) * particle_radius

# CFL condition
velocity_min = -1/2 * particle_radius / delta_time
velocity_max =  1/2 * particle_radius / delta_time

initial_velocity_sampler = torch.distributions.Uniform(velocity_min, velocity_max)
initial_position_sampler = torch.distributions.Uniform(position_min, position_max)

with open(config_file, "w") as f:
    config = {
        "dimention": 3,

        "delta_time"   : delta_time,
        "num_timesteps": 150,

        "rest_density" : 998.29,
        "particle_mass": 0.04,
        "num_neighbor_particles": 30,

        "boundary_size": [20, 20, 20],
        "fluid_size"   : [10, 10, 10],

        "external_force": [0.0, -9.8, 0.0],

        "constraint_relaxation"    : 0.001,
        "allowable_compressibility": 0.05,
        "num_max_iterations"       : 10,
        "k"                        : 0.1,
        "n"                        : 4.0,
        "q"                        : 0.2,
        "c"                        : 0.01,

        "initial_velocity": initial_velocity_sampler.sample((1, 3)).squeeze().tolist(),
        "initial_position": initial_position_sampler.sample((1, 1)).squeeze().tolist()
    }

    json.dump(config, f, indent=4)
