# snnmetrics

This package provides metrics that are specific to spiking neural networks. Currently in beta phase.

## Number of synaptic operations (SynOps)
1. Define a SynOps metric for each spiking layer by providing the fanout as float (mostly used for Linear layers) or tensor with dimensions (C,H,W), mostly used for conv layers.
    ```
    import snnmetrics as sm
    synops_layer1 = sm.SynOps(fanout=10.)
    synops_layer2 = sm.SynOps(fanout=100.)
    ```
2. Get activations of intermediate spiking layers either from model directly or through forward hooks.
    ```
    y_hat, (layer1_activations, layer2_activations) = model(x)
    ```
3. Pass activations to synops metrics to compute batch statistics. Sum over time if necessary, allowed shapes are (B,C) or (B,C,H,W). Batch statistics will be averaged across the batch dimension so you'll likely end up with non-integer synops.
    ```
    batch_stats_layer1 = synops_layer1(layer1_activations)
    synops_per_neuron = batch_stats_layer1['synops_per_neuron']
    synops = batch_stats_layer1['synops']
    ```
4. At the end of the epoch, compute the average synops across all mini-batches.
    ```
    epoch_stats = synops_layer1.compute()
    epoch_synops = epoch_stats['synops']
    ```
5. Before the start of the next epoch, reset the metric.
    ```
    synops_layer1.reset()
    ```

```{toctree}
:hidden:
examples/examples
getting_involved
```