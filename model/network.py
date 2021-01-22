import torch
from torch_geometric.nn import TopKPooling, GCNConv
from torch import nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        # self.in_channels = 3
        # self.pool_neck = 512
        # self.amount_descriptors = 512

        self.activation = nn.LeakyReLU()

        self.conv1 = GCNConv(in_channels=3, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=32)
        self.conv3 = GCNConv(in_channels=32, out_channels=64)

        self.pooling1 = TopKPooling(in_channels=16, ratio=512)
        self.pooling2 = TopKPooling(in_channels=64, ratio=128)

        self.flatten = nn.Flatten(start_dim=0)  # Special start dim as it is not yet batched
        self.fc1 = nn.Linear(64*128, 128)
        #self.fc2 = nn.Linear(2048, 128)


    def forward(self, data):
        pos, edge_index = data.pos, data.edge_index

        x = self.conv1(pos, edge_index)
        x = self.activation(x)
        x, edge_index, edge_attr, batch, perm, score = self.pooling1(x, edge_index)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x, edge_index, edge_attr, batch, perm, score = self.pooling2(x, edge_index)
        x = self.activation(x)

        # Flatten type
        x = x.transpose(0, 1)  # transpose to regular structure of [dims, nodes]
        x = self.flatten(x)
        x = self.fc1(x)
        #x = self.activation(x)
        #x = self.fc2(x)

        return x


# The model described in the reliminary report
# DO NOT USE, as it has a lot of limitations
# Mostly as a test to see if the current re-written code gets the same result
class PrelimNet(torch.nn.Module):
    def __init__(self):
        super(PrelimNet, self).__init__()

        self.activation = nn.LeakyReLU()

        self.conv1 = GCNConv(in_channels=3, out_channels=5)
        self.conv2 = GCNConv(in_channels=5, out_channels=20)

        self.flatten = nn.Flatten(start_dim=0)  # Special start dim as it is not yet batched
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(5850*10, 100)

    def forward(self, data):
        pos, edge_index = data.pos, data.edge_index

        x = self.conv1(pos, edge_index)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.activation(x)

        x = self.fc1(x)
        x = self.activation(x)

        # Flatten type
        x = x.transpose(0, 1)  # transpose to regular structure of [dims, nodes]
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x
