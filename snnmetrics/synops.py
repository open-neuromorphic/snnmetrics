from typing import Optional

import torch
import torch.nn as nn
from torchmetrics.metric import Metric


class SynOps(Metric):
    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self, fanout: torch.Tensor, sample_time: Optional[float] = None):
        super().__init__()
        self.fanout = torch.as_tensor(fanout)
        self.sample_time = sample_time
        self.add_state(
            "synops_per_neuron",
            default=[]
            if self.fanout.shape == torch.Size([])
            else torch.zeros(self.fanout.shape),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: torch.Tensor):
        if self.fanout.shape == torch.Size([]):
            self.synops_per_neuron.append(output.sum(0) * self.fanout)
        else:
            self.synops_per_neuron += output.sum(0) * self.fanout
        self.total += output.shape[0]

    def compute(self):
        if self.fanout.shape == torch.Size([]):
            synops = torch.stack(self.synops_per_neuron).sum(0) / self.total
        else:
            synops = self.synops_per_neuron / self.total
        result_dict = {"synops_per_neuron": synops, "synops": synops.sum()}
        if self.sample_time is not None:
            result_dict["synops/s"] = synops.mean() / self.sample_time
        return result_dict
