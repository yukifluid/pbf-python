import sys
import torch
from simulator import PositionBasedFluids

if len(sys.argv) < 2:
    print("not enough arguments", file=sys.stderr)
    exit(1)

scene_path = sys.argv[1]
config_file = f"{scene_path}/config.json"
output_file = f"{scene_path}/data.csv"
measurement_file = f"{scene_path}/measurement.csv"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

simulator = PositionBasedFluids(device)
simulator.reset(config_file)
simulator.run(output_file, measurement_file)