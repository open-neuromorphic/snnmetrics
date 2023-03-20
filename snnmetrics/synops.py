from typing import Optional, Union

import torch
from torchmetrics.metric import Metric


class SynOps(Metric):
    """A metric that calculates the number of synaptic operations, both for every neuron in the
    layer and for the sum over all neurons in the layer. The number of synaptic operations is
    defined as number of spikes times the fanout, which are the number of connections each neuron
    has to the next layer. Whereas the fanout using fully-connected connectivity is equal to the
    number of neurons (or features) in the next layer, the situation for convolutional layers is
    more complex. Parameters such as stride, kernel size, grouping and others all have influence on
    convolutional fanout. When you think about a convolutional kernel that is applied to every
    receptive field, the neurons at the edge of the input will be seen less often (given a padding
    of zero) than neurons in the middle. The convolutional fanout can be approximated when the
    spatial input size is large enough.

    Parameters:
        fanout: Can either be a float or a tensor of shape (C,H,W).
    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self, fanout: Union[float, torch.Tensor], sample_time: Optional[float] = None
    ):
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
