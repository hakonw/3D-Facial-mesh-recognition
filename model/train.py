import torch
from tqdm import tqdm

import datasetFacegen
from torch_geometric.data import DataLoader
import utils
import metrics

import dataclasses

torch.manual_seed(1)
torch.cuda.manual_seed(1)
import numpy as np
np.random.seed(1)
import random
random.seed(1)
#torch.set_deterministic(True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("MISSING CUDA")
    raise Exception("Missing cuda")

print("Cuda:", torch.cuda.is_available())
print("Type:", device.type)



# Values
from setup import Config as cfg

writer = cfg.WRITER

# Actual code
print("Loading model")
model = cfg.MODEL
model = model.to(device)

# Loss & optimizer
print("Loading criterion and optimizer")
criterion = torch.nn.TripletMarginLoss(margin=cfg.MARGIN, p=cfg.P, reduction=cfg.REDUCTION)  # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss   mean or sum reduction possible
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LR)

print("Loading Dataset")
facegen_helper = datasetFacegen.FaceGenDatasetHelper(root=cfg.DATASET_PATH, pickled=cfg.DATASET_SAVE, face_to_edge=cfg.DATASET_EDGE)
facegen_dataset = datasetFacegen.FaceGenDataset(facegen_helper.get_cached_dataset())
print("Loading DataLoader")
dataloader = DataLoader(dataset=facegen_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

print("Staring")
iter = 0
for epoch in range(cfg.EPOCHS):
    tq = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{cfg.EPOCHS}")

    losses = []
    for i_batch, sample_batched in tq:
        optimizer.zero_grad()

        sample_batched[0].to(device)
        sample_batched[1].to(device)

        # Sample_batched contains pairs in the format [2, batch_size]
        sample_1 = sample_batched[0].to_data_list()
        sample_2 = sample_batched[1].to_data_list()

        descritors = []
        for i in range(len(sample_1)):
            desc1 = model(sample_1[i])
            desc2 = model(sample_2[i])
            descritors.append((desc1, desc2))
        anchors, positives, negatives = utils.findpairs(descritors, accept_all=cfg.ALL_TRIPLETS)

        # loss
        loss = criterion(anchors, positives, negatives)

        iter += 1
        losses.append(loss.item())
        writer.add_scalar('Loss/train', loss.item(), iter)
        writer.add_scalar('Pairs/train', len(anchors), iter)
        if iter % 5 == 0:
            tq.set_postfix(avg_loss=sum(losses)/max(len(losses), 1), pairs=len(anchors))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Metrics
    if (epoch + 1) % cfg.EPOCH_PER_METRIC == 0:
        descriptor_dict = metrics.data_dict_to_descriptor_dict(model=model, device=device, data_dict=facegen_helper.get_cached_dataset(), desc="Evaluation/Test", leave_tqdm=False)
        metric = metrics.get_metric_all_vs_all(margin=1.0, descriptor_dict=descriptor_dict)
        print(metric)
        metric_dict = dataclasses.asdict(metric)
        for metric_key in metric_dict.keys():
            writer.add_scalar("metric-" + metric_key + "/train", metric_dict[metric_key], iter)




    writer.add_scalar('AverageEpochLoss/train', sum(losses)/len(losses), epoch)
    writer.flush()

print("Beginning metrics")
descriptor_dict = metrics.data_dict_to_descriptor_dict(model=model, device=device, data_dict=facegen_helper.get_cached_dataset())
print(metrics.get_metric_all_vs_all(margin=1.0, descriptor_dict=descriptor_dict))

# Create embeddings plot
labels = []
features = []
for id, desc_list in descriptor_dict.items():
    for desc in desc_list:
        labels.append(id)
        features.append(desc)
embeddigs = torch.stack(features)
writer.add_embedding(mat=embeddigs, metadata=labels, tag=cfg.run_name)


# Close tensorboard
writer.close()