from torchmetrics.metric import Metric
import torch.nn as nn
import torch
from typing import Optional


class SynOps(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: bool = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = None

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False
    def __init__(self, connect_map: torch.Tensor): # model: nn.Module):
        super().__init__()
        self.add_state("synops_per_neuron", default=torch.zeros(connect_map.shape), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.connect_map = connect_map

    def update(self, output: torch.Tensor):
        self.synops_per_neuron += output.sum(0) * self.connect_map
        self.total += output.shape[0]

    def compute(self):
        return self.synops_per_neuron / self.total