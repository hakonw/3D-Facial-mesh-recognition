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
criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction="mean") # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss   mean or sum reduction possible
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



epochs = 20
batch_size = 4
split = (0.6, 0.4)
train_dataset_size = 100*split[0]
batch_iter_update = (train_dataset_size/batch_size)/8  # Update 8 times per epoch

# Get the FaceGen dataset
faceGenDataset = FaceGenDataset(device="cpu")  # Must load it into cpu memory atm, see Multi-process data loading https://pytorch.org/docs/stable/data.html # TODO load the entire dataset into ram?

# Split into train and val
lengths = [int(len(faceGenDataset)*split[0]), int(len(faceGenDataset)*split[1])]
dataset_train, dataset_val = random_split(faceGenDataset, lengths)
print(f"Dataset loaded, split: {split[0]}/{split[1]}")

# Create an evaluator to validate accuracy
evaluator = Evaluation(dataset=dataset_val, margin=1.0, device=device)

# Get the train dataloader ready
dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
print("Dataloader ready")

losses = {}
losses_avg = {}
try:
    print("Starting")
    iter = 0  # number_of_samples / batch_size per epoch
    for epoch in range(epochs):

        running_loss = 0
        loss_samples = 0

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

            running_loss += loss.item()
            loss_samples += 1
            iter += 1  # +1 before check, as the last element in the list should have the correct loss
            losses[iter] = loss.item()
            # batch_iter_size * i_batch + total*epoch/batch ?
            if iter % batch_iter_update == 0:
                tq.set_postfix(loss=loss.item(), avg_loss=running_loss/loss_samples)



            optimizer.zero_grad()
            #a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            #print(list(model.parameters())[0].grad)
            #b = list(model.parameters())[0].clone()
            #print(torch.equal(a.data, b.data))
        losses_avg[epoch] = running_loss/loss_samples

        #evaluator(model=model)
except KeyboardInterrupt:
    evaluator(model=model)
    print("bye")
    import sys
    sys.exit()

evaluator(model=model)
print("Evaluating n_vs_nn")
evaluator(model=model, n_vs_nn=True)
#print(losses)
import matplotlib.pylab as plt
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.ylim(-0.1, 2)
plt.plot(*zip(*sorted(losses.items())), label="Loss")
plt.show()
plt.savefig("loss.png")
plt.close()

plt.xlabel("Epoch")
plt.ylabel("Average loss")
plt.ylim(-0.1, 1.25)
plt.plot(*zip(*sorted(losses_avg.items())), label="Average loss")
plt.show()
plt.savefig("loss-avg.png")


if __name__ == "__main__":
    pass
