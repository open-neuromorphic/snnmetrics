import torch

import snnmetrics


def test_synops_conv_output():
    fanout = torch.eye(3).unsqueeze(0)
    output = torch.ones((2, 1, 3, 3))
    output2 = torch.ones((2, 1, 3, 3)) * 2

    metric = snnmetrics.SynOps(fanout=fanout)

    batch1_synops = metric(output)
    assert (batch1_synops["synops_per_neuron"] == fanout).all()
    assert batch1_synops["synops"] == 3.0

    batch2_synops = metric(output2)
    assert (batch2_synops["synops_per_neuron"] == fanout * 2).all()
    assert batch2_synops["synops"] == 6.0

    epoch_synops = metric.compute()
    assert (epoch_synops["synops_per_neuron"] == fanout * 1.5).all()
    assert epoch_synops["synops"] == 4.5


def test_synops_conv_output_float_fanout():
    fanout = torch.tensor(3)
    output = torch.ones((2, 1, 3, 3))
    output2 = torch.ones((2, 1, 3, 3)) * 2

    metric = snnmetrics.SynOps(fanout=fanout)

    batch1_synops = metric(output)
    assert (batch1_synops["synops_per_neuron"] == output.mean(0) * fanout).all()
    assert batch1_synops["synops"] == (output.mean(0) * fanout).sum()

    batch2_synops = metric(output2)
    assert (batch2_synops["synops_per_neuron"] == output2.mean(0) * fanout).all()
    assert batch2_synops["synops"] == (output2.mean(0) * fanout).sum()

    epoch_synops = metric.compute()
    assert (
        epoch_synops["synops_per_neuron"]
        == (output.mean(0) + output2.mean(0)) * fanout / 2
    ).all()
    assert (
        epoch_synops["synops"]
        == ((output.mean(0) + output2.mean(0)) * fanout / 2).sum()
    )


def test_synops_linear_output():
    fanout = torch.tensor(100)
    batch_size = 2
    output = torch.ones((batch_size, 10))
    output2 = torch.ones((batch_size, 10)) * 2

    metric = snnmetrics.SynOps(fanout=fanout)

    batch1_synops = metric(output)
    assert (
        batch1_synops["synops_per_neuron"] == output.sum(0) * fanout / batch_size
    ).all()

    batch2_synops = metric(output2)
    assert (
        batch2_synops["synops_per_neuron"] == output2.sum(0) * fanout / batch_size
    ).all()

    epoch_synops = metric.compute()
    assert (
        epoch_synops["synops"]
        == (output.sum() + output2.sum()) * fanout / batch_size / 2
    )
    assert (
        epoch_synops["synops_per_neuron"]
        == (output.sum(0) + output2.sum(0)) * fanout / batch_size / 2
    ).all()
