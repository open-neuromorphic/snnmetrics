import snnmetrics
import torch


def test_synops():
    connect_map = torch.eye(3)

    output = torch.ones((2,3,3))
    output2 = torch.ones((2,3,3)) * 2

    metric = snnmetrics.SynOps(connect_map=connect_map)
    
    batch1_synops = metric(output)
    assert (batch1_synops == connect_map).all()

    batch2_synops = metric(output2)
    assert (batch2_synops == connect_map*2).all()

    epoch_synops = metric.compute()
    assert (epoch_synops == connect_map * 1.5).all()
