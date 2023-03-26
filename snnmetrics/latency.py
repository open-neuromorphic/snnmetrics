from typing import Optional, Union

import torch
from torchmetrics.metric import Metric


class Latency(Metric):
    """Calculates accuracy over time for a sequential output. Input shape must be (Batch, Time,
    Channel)

    Args:
        prediction: Choose if the prediction is calculated on the sum or the maximum activity over time.
    """

    def __init__(self, pred_type: str = "sum"):
        super().__init__()
        self.pred_type = pred_type
        assert pred_type in ["sum", "max"]
        self.add_state(
            "correct_pred_over_time",
            default=[],
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: torch.Tensor, targets: torch.Tensor):
        assert len(output.shape) == 3
        if self.pred_type == "sum":
            accumulated_spikes = torch.cumsum(input=output, dim=1)
            pred_over_time = accumulated_spikes.argmax(2)
            targets_over_time = targets.unsqueeze(1).repeat(1, output.shape[1])
            self.correct_pred_over_time = (
                (pred_over_time == targets_over_time).float().mean(0)
            )
        self.total += output.shape[0]

    def compute(self):
        return self.correct_pred_over_time / self.total
