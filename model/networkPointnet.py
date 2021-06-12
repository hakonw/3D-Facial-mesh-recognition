import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Flatten, BatchNorm1d as BN
# from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        # print("x", x.shape)
        # print("pos", pos.shape)
        # print("batch", batch.shape)
        # print("pos idk", pos[idx].shape)
        # print("edge", edge_index.shape)
        # print("self.ratio", self.ratio)
        # print("self.r", self.r)
        # print("conv", self.conv)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(1)

        self.sa1_module = SAModule(ratio=0.5, r=0.2, nn=MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(ratio=0.25, r=0.4, nn=MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 256)
        self.flatten = Flatten(start_dim=0)  # Special start dim as it is not yet batched
        self.flatten_batched = Flatten()

    def forward(self, data):
        batch_mode = False
        if isinstance(data, Batch):
            batch_mode = True
            sa0_out = (None, data.pos, data.batch)  # x pos batch
        elif isinstance(data, Data):
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
            sa0_out = (None, data.pos, batch)  # x pos batch
        else:
            raise RuntimeError(f"Illegal data of type: {type(data)}")

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        # print(x.shape)
        x = self.lin3(x)
        if batch_mode:
            x = self.flatten_batched(x)
        else:
            x = self.flatten(x)  # Spesial flatten
        return x
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)

    def short_rep(self):
        modules = self._modules
        out_str = ""  # Semi efficient
        for name, type in modules.items():
            str_rep = str(type)
            if "LeakyReLU" in str_rep:
                str_rep = "LeakyReLU"
            str_rep = str_rep.replace(", bias=True", "")
            out_str += str_rep + ", "
        out_str = out_str[:-2]  # Remove ", " at the end
        return out_str


# def train(epoch):
#     model.train()
#
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         loss = F.nll_loss(model(data), data.y)
#         loss.backward()
#         optimizer.step()
#
#
# def test(loader):
#     model.eval()
#
#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         with torch.no_grad():
#             pred = model(data).max(1)[1]
#         correct += pred.eq(data.y).sum().item()
#     return correct / len(loader.dataset)


# if __name__ == '__main__':
#     path = osp.join(
#         osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
#     pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
#     train_dataset = ModelNet(path, '10', True, transform, pre_transform)
#     test_dataset = ModelNet(path, '10', False, transform, pre_transform)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
#                               num_workers=6)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
#                              num_workers=6)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Net().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     for epoch in range(1, 201):
#         train(epoch)
#         test_acc = test(test_loader)
#         print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))