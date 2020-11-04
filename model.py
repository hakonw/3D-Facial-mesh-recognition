import torch
from torch import nn
from pytorch3d import ops
import pytorch3d
from pytorch3d.structures import Meshes



class TestNet(torch.nn.Module):
    def __init__(self, device, debug=False):
        super(TestNet, self).__init__()
        self.device = device
        self.debug = debug

        self.conv1 = pytorch3d.ops.GraphConv(input_dim=3, output_dim=5, init="normal", directed=False)
        self.conv2 = pytorch3d.ops.GraphConv(input_dim=5, output_dim=20, init="normal", directed=False)

        self.activation = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        siste_lag = 10
        self.fc1 = nn.Linear(20, siste_lag)
        self.fc2 = nn.Linear(5850*siste_lag, 64)  # 5850, as thats the current max edges, TODO fix better way

        self.softmax = nn.Softmax(dim=1)


    def forward(self, input: Meshes):

        # Pack the vertexs and adges
        verts = input.verts_packed().to(self.device)
        edges = input.edges_packed().to(self.device)

        if self.debug: print("vertes", verts.shape)
        if self.debug: print("edges", edges.shape)

        x = self.conv1(verts, edges)
        x = self.activation(x)

        x = self.conv2(x, edges)
        x = self.activation(x)

        #x = self.conv3(x, edges)
        #x = self.activation(x)


        # Split into alle batches?
        # Så litt cheaky dense for å redusere dims ned til det vi vil ha
        # for å så returne [antall i batches, antall siste dim] ?
        # Alt bruk mesh.getitem, så split TODO se på om bedre
        verts_per_mesh = input.num_verts_per_mesh()
        list_of_tensors = torch.split(x, verts_per_mesh.tolist())
        # VIL FUCKE OPP OM ANDRE STØRRELSER. TODO pad?, aka med pytorch3d?
        splitted = torch.stack(list_of_tensors)
        if self.debug: print("splitted?", splitted.shape)

        x = self.fc1(splitted)
        x = self.activation(x)

        x = self.flatten(x)
        x = self.fc2(x)
        #x = self.activation(x)
        x = self.softmax(x)

        return x