import torch

class Box:
    def __init__(self, b_size: torch.Tensor, r: float) -> None:

        self._clamp_min = -(b_size - 2) * r
        self._clamp_max =  (b_size - 2) * r

    def respond(self, p: torch.Tensor) -> torch.Tensor:
        clamped_p = p.clamp(self._clamp_min, self._clamp_max)
        return clamped_p