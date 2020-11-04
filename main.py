import torch
import pytorch3d
from pytorch3d.io import load_ply, load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
#mport matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    #raise Exception("Missing cuda")
    print("MISSING CUDA")
#device = torch.device("cuda:0")
print(torch.cuda.is_available())


#from pytorch3d.renderer import TexturesVertex, OpenGLPerspectiveCameras, PointLights, SoftPhongShader, TexturesUV

#   _______ ______          _____   ____ _______
#  |__   __|  ____|   /\   |  __ \ / __ \__   __|
#     | |  | |__     /  \  | |__) | |  | | | |
#     | |  |  __|   / /\ \ |  ___/| |  | | | |
#     | |  | |____ / ____ \| |    | |__| | | |
#     |_|  |______/_/    \_\_|     \____/  |_|
#
#
verts, faces = load_ply("teapot.ply")  # (V,3) tensor, (F,3) tensor
#model_textures = TexturesVertex(verts_features=torch.ones_like(verts, device=device)[None])
mesh = Meshes(verts=[verts], faces=[faces])#, textures=model_textures)


#from PIL import Image
#import numpy as np
#im = Image.open("001.bmp")
#p = np.array(im)

#   ______            __
#  |  ____|          /_ |
#  | |__ __ _  ___ ___| |
#  |  __/ _` |/ __/ _ \ |
#  | | | (_| | (_|  __/ |
#  |_|  \__,_|\___\___|_|
#
verts, faces, aux = load_obj("001.obj")
#model_textures = TexturesVertex(verts_features=torch.ones_like(verts, device=device)[None])
mesh_face = Meshes(verts=[verts], faces=[faces.verts_idx])#, textures=model_textures)




# Trash, ikke prøv deg lol
#model_textures_uv = TexturesUV(maps = p, faces_uvs=faces.textures_idx, verts_uvs=aux.verts_uvs)
#mesh = load_objs_as_meshes(["001.obj"], device=device)
#texture_image=mesh.textures.maps_padded()


# # Imports
# from pytorch3d.renderer import (
#     FoVPerspectiveCameras, look_at_view_transform,
#     RasterizationSettings, BlendParams,
#     MeshRenderer, MeshRasterizer, HardPhongShader, SoftSilhouetteShader, Materials, TexturesVertex
# )
# R, T = look_at_view_transform(2.7, 10, 20)
# cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
#
# raster_settings = RasterizationSettings(
#     image_size=512,
#     blur_radius=0.0,
#     faces_per_pixel=1,
# )
#
# #R, T = look_at_view_transform(8.0, -80, 10)
# R, T = look_at_view_transform(80, 80, 80)
# cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
# raster_settings = RasterizationSettings(image_size=512)
# lights = PointLights(location=torch.tensor([200.0, -20, 20], device=device)[None],device=device)
#
# renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#     shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
# )


#images = renderer(mesh)
#plt.figure(figsize=(10, 10))
#plt.imshow(images[0, ..., :3].cpu().numpy())
#plt.grid("off")
#plt.axis("off")
#plt.show()


import torch
import model

print(torch.cuda.is_available())
print(device.type)

#
# Meshes 1-3
#
mesh_multiple = load_objs_as_meshes(["001.obj", "002.obj", "003.obj"], device=device)

model = model.TestNet(device)
model = model.to(device)  #  a must

#criterion = torch.nn.MSELoss(reduction='sum')
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

#mesh = mesh.cuda()
mesh = mesh_multiple.to(device)
print("verts to mesh idx:", mesh.verts_packed_to_mesh_idx())
print("verts to mesh idx:", mesh.verts_packed_to_mesh_idx().shape)
print("Edge to mesh idx:", mesh.edges_packed_to_mesh_idx())
print("Edge to mesh idx:", mesh.edges_packed_to_mesh_idx().shape)
print("num verts per mesh:", mesh.num_verts_per_mesh())
print("num Edges per mesh:", mesh.num_edges_per_mesh())

#print("faces to mesh idx:", mesh.faces_packed_to_mesh_idx())
#print("faces to mesh idx:", mesh.faces_packed_to_mesh_idx().shape)
#print("faces to edges packed:", mesh.faces_packed_to_edges_packed().shape)

#y_pred = model(mesh)

#y_hat = torch.tensor([0,1,2]).to(device)
y_hat = torch.tensor([[0.0]*64, [0.0]*64, [0.0]*64,]).to(device)
y_hat[0][0] = 1.0
y_hat[1][1] = 1.0
y_hat[2][2] = 1.0

from tqdm import tqdm
tq = tqdm(range(5000), leave=False)

losses = {}

for t in tq:
    #print("t", t)


    y_pred = model(mesh)
    #print("y_pred", y_pred.shape)
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