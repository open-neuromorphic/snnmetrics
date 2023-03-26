import torch

import snnmetrics


def test_latency():
    batch_size, n_steps, n_classes = 5, 10, 4
    output = torch.zeros((batch_size, n_steps, n_classes))
    y = torch.randint(0, n_classes, size=(batch_size,))
    first_spikes = torch.randint(0, n_steps, size=(batch_size,))
    for i, (latency, target) in enumerate(zip(first_spikes, y)):
        output[i, latency:, target] = 1
        if target == 0:
            output[i, :, -1] = 0.05

    metric = snnmetrics.Latency("sum")
    latency = metric(output, y)

    true_latency = torch.zeros_like(latency)
    for time in first_spikes:
        true_latency[time] += 1
    true_latency = true_latency.cumsum(0) / batch_size

    assert (true_latency == latency).all()
