import torch
import model
from tqdm import tqdm
from facegenDataset import FaceGenDataset
from torch.utils.data import DataLoader, random_split
from evaluation import Evaluation

from pytorch3d.structures import Meshes

torch.manual_seed(1)
import numpy as np
np.random.seed(1)  # numpy might be used in a library, force seed there too
torch.cuda.manual_seed(1)
import random
random.seed(1)
torch.set_deterministic(True)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("MISSING CUDA")
    raise Exception("Missing cuda")

print("Cuda:", torch.cuda.is_available())
print("Type:", device.type)


# Model
model = model.TestNet(device)
model = model.to(device)  # a must


# Loss & optimizer
# criterion = torch.nn.MSELoss(reduction='sum')
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)  # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


losses = {}
epochs = 20
batch_size = 4
batch_iter_size = 80/batch_size  # TODO move to use len facegendataset

# Get the FaceGen dataset
faceGenDataset = FaceGenDataset(device="cpu")  # Must load it into cpu memory atm, see Multi-process data loading https://pytorch.org/docs/stable/data.html # TODO load the entire dataset into ram?

# Split into train and val
lengths = [int(len(faceGenDataset)*0.6), int(len(faceGenDataset)*0.4)]
dataset_train, dataset_val = random_split(faceGenDataset, lengths)
print("Dataset loaded")

# Create an evaluator to validate accuracy
evaluator = Evaluation(dataset=dataset_val, margin=1.0, device=device)

# Get the train dataloader ready
dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
print("Dataloader ready")


try:

    print("Starting")
    for epoch in range(epochs):

        tq = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}/{epochs}")
        for i_batch, sample_batced in tq:

            # Får 4 (batch size) rader i verts, og verts_idx
            # Atm:
            #   Første rad: Anchor
            #   Første rad alt: Positive
            #   Rad 2-4: Negative
            #   Rad 2-4 alt: Negative
            # Dupe anchor positive, for alle negative
            verts_reg = sample_batced["regular"]["verts"]
            verts_reg_idx = sample_batced["regular"]["verts_idx"]
            verts_alt = sample_batced["alt"]["verts"]
            verts_alt_idx = sample_batced["alt"]["verts_idx"]

            # Generate multiple Meshes
            verts = torch.cat([verts_reg, verts_alt], dim=0)
            verts_idx = torch.cat([verts_reg_idx, verts_alt_idx], dim=0)
            meshes = Meshes(verts=verts, faces=verts_idx)
            meshes.to(device)

            y_pred = model(meshes)

            y_pos = y_pred[0].repeat((batch_size*2)-2, 1)
            y_anchor = y_pred[batch_size].repeat((batch_size*2)-2, 1)
            y_negative = torch.cat([y_pred[1:batch_size], y_pred[batch_size+1:]], dim=0)
            #assert torch.all(torch.eq(y_pos[1], y_pos[2]))

            loss = criterion(y_pos, y_anchor, y_negative)

            if i_batch % batch_iter_size == 0:
                losses[(epoch*batch_iter_size) + i_batch] = loss.item()
                tq.set_postfix(loss=loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #evaluator(model=model)
except KeyboardInterrupt:
    evaluator(model=model)
    print("bye")
    import sys
    sys.exit()

evaluator(model=model)
print(losses)
import matplotlib.pylab as plt
plt.plot(*zip(*sorted(losses.items())))
plt.show()




if __name__ == "__main__":
    pass
