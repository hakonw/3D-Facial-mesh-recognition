import torch
from tqdm import tqdm
import network
import datasetFacegen
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1)
torch.cuda.manual_seed(1)
#import numpy as np
#np.random.seed(1)
#import random
#random.seed(1)
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
writer = SummaryWriter("log", comment="aaa")
# writer.add_scalar('Loss/train', np.random.random(), n_iter)

# Values
epochs = 10
batch_size = 1


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
    for i_batch, sample_batched in tq:

        sample_batched[0].to(device)
        sample_batched[1].to(device)

        # Sample_batched contains pairs in the format [2, batch_size]
        sample_anc = sample_batched[0].to_data_list()
        sample_pos = sample_batched[1].to_data_list()


        descritors = []
        for sample in sample_anc + sample_pos:
            descritors.append(model(sample))


        iter += 1



        # Pairs split the following way:
        #   first [batch_size] of anchors, then [batch_size] of positive



# Close tensorboard
writer.close()