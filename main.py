import torch
from pytorch3d.io import load_ply, load_objs_as_meshes, load_obj
#mport matplotlib.pyplot as plt
import model
from tqdm import tqdm
from facegenDataset import FaceGenDataset
from torch.utils.data import DataLoader

from pytorch3d.structures import Meshes

torch.manual_seed(1)
import numpy as np
np.random.seed(1)  # numpy might be used in a library, force seed there too

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("MISSING CUDA")
    raise Exception("Missing cuda")

print(torch.cuda.is_available())
print(device.type)


# from pytorch3d.renderer import TexturesVertex, OpenGLPerspectiveCameras, PointLights, SoftPhongShader, TexturesUV

#   _______ ______          _____   ____ _______
#  |__   __|  ____|   /\   |  __ \ / __ \__   __|
#     | |  | |__     /  \  | |__) | |  | | | |
#     | |  |  __|   / /\ \ |  ___/| |  | | | |
#     | |  | |____ / ____ \| |    | |__| | | |
#     |_|  |______/_/    \_\_|     \____/  |_|
#
#
# verts, faces = load_ply("teapot.ply")  # (V,3) tensor, (F,3) tensor
# model_textures = TexturesVertex(verts_features=torch.ones_like(verts, device=device)[None])
# mesh = Meshes(verts=[verts], faces=[faces])#, textures=model_textures)


#   ______            __
#  |  ____|          /_ |
#  | |__ __ _  ___ ___| |
#  |  __/ _` |/ __/ _ \ |
#  | | | (_| | (_|  __/ |
#  |_|  \__,_|\___\___|_|
#
# verts, faces, aux = load_obj("001.obj")
# model_textures = TexturesVertex(verts_features=torch.ones_like(verts, device=device)[None])
# mesh_face = Meshes(verts=[verts], faces=[faces.verts_idx])#, textures=model_textures)



#
# Meshes 1-3
#
#mesh_multiple = load_objs_as_meshes(["001.obj", "002.obj", "003.obj"], device=device)


# Model
model = model.TestNet(device)
model = model.to(device)  # a must


# Loss & optimizer
# criterion = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Mess debug info
# mesh = mesh_multiple.to(device)
# print("verts to mesh idx:", mesh.verts_packed_to_mesh_idx())
# print("verts to mesh idx:", mesh.verts_packed_to_mesh_idx().shape)
# print("Edge to mesh idx:", mesh.edges_packed_to_mesh_idx())
# print("Edge to mesh idx:", mesh.edges_packed_to_mesh_idx().shape)
# print("num verts per mesh:", mesh.num_verts_per_mesh())
# print("num Edges per mesh:", mesh.num_edges_per_mesh())

# print("faces to mesh idx:", mesh.faces_packed_to_mesh_idx())
# print("faces to mesh idx:", mesh.faces_packed_to_mesh_idx().shape)
# print("faces to edges packed:", mesh.faces_packed_to_edges_packed().shape)

# Generate label data for testing
#y_hat = torch.tensor([0,1,2,0,0,0,0,0,0,0]).to(device)
#y_hat = torch.tensor([[0.0]*64, [0.0]*64, [0.0]*64,]).to(device)
#y_hat[0][0] = 1.0
#y_hat[1][1] = 1.0
#y_hat[2][2] = 1.0


losses = {}
epochs = 10

faceGenDataset = FaceGenDataset(device="cpu") # Must load it into cpu memory atm, see Multi-process data loading https://pytorch.org/docs/stable/data.html
print("Dataset loaded")
dataloader = DataLoader(dataset=faceGenDataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
print("Dataloader ready")

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

        y_hat_py = [int(i)-1 for i in sample_batced["idd"] + sample_batced["idd"]]
        y_hat = torch.tensor(y_hat_py, dtype=torch.long, device=device)

        y_pred = model(meshes)
        # print("y_pred", y_pred.shape)
        loss = criterion(y_pred, y_hat)

        if i_batch % 25 == 0:
            losses[(epoch*25) + i_batch] = loss.item()
            tq.set_postfix(loss=loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(losses)
import matplotlib.pylab as plt
plt.plot(*zip(*sorted(losses.items())))
plt.show()

if __name__ == "__main__":
    pass