from torch.utils.tensorboard import SummaryWriter
import torch_geometric.nn as nn
import os
import dataclasses

import setup
import train
import network



# To try different things

# Conv algorithms
# TODO add the rest
conv_algos = [nn.GCNConv, nn.SAGEConv, nn.GraphConv, nn.GATConv, nn.TAGConv, nn.ARMAConv, nn.ClusterGCNConv, nn.FeaStConv, nn.GENConv, nn.SGConv]
# nn.ChebConv  disabled due to needing a "K", filter size
# nn.GravNetConv  due to 'space_dimensions', 'propagate_dimensions', and 'k'
# nn.AGNNConv  doesnt have any in or out channel

# GraphConv, ASAPooling -- Failing triplets? What, and nan loss

# Pooling algorithms
pool_algos = [nn.TopKPooling, nn.SAGPooling, nn.ASAPooling]

log_dir = "experiment/"
try:
    previous_runs = os.listdir(log_dir)
    if len(previous_runs) == 0:
        run_number = 0
    else:
        previous_runs = list(filter(lambda x: "experiment" in x, previous_runs))
        run_number = max([int(s.split('experiment_')[1]) for s in previous_runs]) + 1
except FileNotFoundError:
    run_number = 0
run_name = f"experiment_{run_number:03}"
writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name), max_queue=20)
print("Beginning experiments", run_name)

iter = 0
max_iter = len(pool_algos) * len(conv_algos)
for pool_algo in pool_algos:
    for conv_algo in conv_algos:

        print(f"Beginning experiment {iter}/{max_iter} with {conv_algo.__name__}, {pool_algo.__name__}")
        config = setup.Config(enable_writer=False)  # Default config, to be modified different things

        config.MODEL = network.TestNetTryMode(conv=conv_algo, pool=pool_algo)

        config.TENSORBOARD_EMBEDDINGS = False
        config.TENSORBOARD_HPARAMS = False
        config.TENSORBOARD_TEXT = False
        config.WRITE_MODEL_SUMMARY = False
        config.LAVE_TQDM = False

        config.EPOCH_PER_METRIC = 40

        last_metric = train.train(config)

        metric_dict = dataclasses.asdict(last_metric)
        newdict = {}
        for metric_key in metric_dict.keys():
            newdict["hparam-experiment-" + metric_key + "/train"] = metric_dict[metric_key]
        writer.add_hparams(
            hparam_dict={
                "Model": str(config.MODEL.__class__.__name__),
                "Convolution": conv_algo.__name__,
                "Pooling": pool_algo.__name__,
              },
            metric_dict=dataclasses.asdict(last_metric),
            run_name=f"experiment{iter}"
         )

        iter += 1



