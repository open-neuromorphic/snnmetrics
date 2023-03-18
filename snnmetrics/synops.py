from typing import Optional

import torch
import torch.nn as nn
from torchmetrics.metric import Metric


class SynOps(Metric):
    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self, connect_map: torch.Tensor):  # model: nn.Module):
        super().__init__()
        self.add_state(
            "synops_per_neuron",
            default=torch.zeros(connect_map.shape),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.connect_map = connect_map

    def update(self, output: torch.Tensor):
        self.synops_per_neuron += output.sum(0) * self.connect_map
        self.total += output.shape[0]

    def compute(self):
        return self.synops_per_neuron / self.total
