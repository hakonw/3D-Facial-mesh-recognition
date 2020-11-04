import torch
from pytorch3d.io import load_ply, load_objs_as_meshes, load_obj
#mport matplotlib.pyplot as plt
import model
from tqdm import tqdm


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
mesh_multiple = load_objs_as_meshes(["001.obj", "002.obj", "003.obj"], device=device)


# Model
model = model.TestNet(device)
model = model.to(device)  # a must


# Loss & optimizer
# criterion = torch.nn.MSELoss(reduction='sum')
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Mess debug info
mesh = mesh_multiple.to(device)
print("verts to mesh idx:", mesh.verts_packed_to_mesh_idx())
print("verts to mesh idx:", mesh.verts_packed_to_mesh_idx().shape)
print("Edge to mesh idx:", mesh.edges_packed_to_mesh_idx())
print("Edge to mesh idx:", mesh.edges_packed_to_mesh_idx().shape)
print("num verts per mesh:", mesh.num_verts_per_mesh())
print("num Edges per mesh:", mesh.num_edges_per_mesh())

# print("faces to mesh idx:", mesh.faces_packed_to_mesh_idx())
# print("faces to mesh idx:", mesh.faces_packed_to_mesh_idx().shape)
# print("faces to edges packed:", mesh.faces_packed_to_edges_packed().shape)

# Generate label data for testing
#y_hat = torch.tensor([0,1,2]).to(device)
y_hat = torch.tensor([[0.0]*64, [0.0]*64, [0.0]*64,]).to(device)
y_hat[0][0] = 1.0
y_hat[1][1] = 1.0
y_hat[2][2] = 1.0


tq = tqdm(range(5000), leave=False)

losses = {}

for t in tq:

    y_pred = model(mesh)
    # print("y_pred", y_pred.shape)
    loss = criterion(y_pred, y_hat)

    if t % 100 == 0:
        #print(t, loss.item())
        losses[t] = loss.item()
        tq.set_postfix(loss=loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(losses)

if __name__ == "__main__":
    pass