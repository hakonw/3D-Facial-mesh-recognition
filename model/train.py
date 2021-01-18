import torch
from tqdm import tqdm

import datasetFacegen
from torch_geometric.data import DataLoader
import utils

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

        anchors, positives, negatives = utils.findpairs(descritors)

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

    writer.add_scalar('AverageEpochLoss/train', sum(losses)/len(losses), epoch)
    writer.flush()


# Close tensorboard
writer.close()