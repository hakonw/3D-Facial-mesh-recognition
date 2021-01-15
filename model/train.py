import torch
from tqdm import tqdm
import network
import datasetFacegen
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

# https://pytorch.org/docs/stable/tensorboard.html
import os
try:
    previous_runs = os.listdir('log/')
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
except FileNotFoundError:
    run_number = 0
run_name = f"run_{run_number:03}"
writer = SummaryWriter(log_dir=os.path.join("log", run_name), max_queue=20)
print("Beginning", run_name)


# Values
epochs = 40
batch_size = 10


# Actual code
print("Loading model")
model = network.TestNet()
model = model.to(device)

# Loss & optimizer
print("Loading criterion and optimizer")
criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction="mean")  # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss   mean or sum reduction possible
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

print("Loading Dataset")
facegen_helper = datasetFacegen.FaceGenDatasetHelper()
facegen_dataset = datasetFacegen.FaceGenDataset(facegen_helper.get_cached_dataset())
print("Loading DataLoader")
dataloader = DataLoader(dataset=facegen_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print("Staring")
iter = 0
for epoch in range(epochs):
    tq = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{epochs}")

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
#writer.close()