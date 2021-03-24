import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
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

        self.sa1_module = SAModule(0.5, 0.1, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        #self.sa1_module = SAModule(0.5, 0.1, MLP([3, 64, 64, 128]))  # 512 points
        #self.sa2_module = SAModule(0.25, 0.2, MLP([128 + 3, 128, 128, 256])) # 128 points
        #self.sa3_module = SAModule(0.25, 0.4, MLP([256 + 3, 256])) # 128 points
        #self.sa4_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # self.sa1_module = SAModule(ratio=0.5, r=0.2, nn=MLP([3, 64, 64]))
        # self.sa2_module = SAModule(ratio=0.5, r=0.4, nn=MLP([64 + 3, 128, 512]))
        # self.sa3_module = GlobalSAModule(MLP([512 + 3, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 256)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        #sa4_out = self.sa4_module(*sa3_out)
        
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        # x = F.normalize(x, dim=-1, p=2)  # Normalization tips
        return x
        # return F.log_softmax(x, dim=-1)



def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__2':
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))


import torch_geometric.data.batch as geometric_batch
import onlineTripletLoss
def train2(epoch, model, device, dataloader, optimizer, margin):
    model.train()

    losses = []
    dist_a_p = []
    dist_a_n = []
    lengths = []
    for batch in dataloader:
        # create single batch object
        datas = []
        for b in batch:
            datas += b.to_data_list()
        batch_all = geometric_batch.Batch.from_data_list(datas)

        batch_all = batch_all.to(device)
        optimizer.zero_grad()
        descritors = model(batch_all)

        # optimizer.zero_grad()
        # descriptors = [] 
        # for b in batch:
        #     # Each b is for an identity
        #     b = b.to(device)
        #     print(b)
        #     descriptors.append(model(b).to("cpu"))
        # print(descriptors)

        # Create dict again
        dic_descriptors = {}
        for i in range(len(batch_all.id)):
            id = batch_all.id[i].item()
            if id in dic_descriptors:
                dic_descriptors[id].append(descritors[i])
            else:
                dic_descriptors[id] = [descritors[i]]
        descritors = list(dic_descriptors.values())

        #loss = F.nll_loss(model(data), data.y)
        all = []
        labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
        for ident, listt in enumerate(descritors):
            all += listt
            labels += [ident] * len(listt)
        
        all = torch.stack(all).to("cpu")
        labels = torch.tensor(labels).to("cpu")
        loss, fraction_positive_triplets = onlineTripletLoss.batch_all_triplet_loss(labels=labels, embeddings=all, margin=margin)
        # loss = onlineTripletLoss.batch_hard_triplet_loss(labels=labels, embeddings=all, margin=margin)
    
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            losses.append(loss.item())
            dist_a_p.append(torch.dist(descritors[0][0].to("cpu"), descritors[0][1].to("cpu"), p=2).item())
            dist_a_n.append(torch.dist(descritors[1][0].to("cpu"), descritors[0][0].to("cpu"), p=2).item())
            lengths.append(torch.norm(descritors[0][0], 2))
    return losses, dist_a_p, dist_a_n, lengths
    
        

if __name__ == '__main__':
    from setup import Config
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cfg = Config(enable_writer=False)

    dataloader = DataLoader(dataset=cfg.DATASET, batch_size=15, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # loss = ||ap|| - ||an|| + margin.  neg loss => ||an|| >>> ||ap||, at least margin over

    margin = 0.3


    for epoch in range(1, 201):
        losses, dist_a_p, dist_a_n, lengths = train2(epoch, model, device, dataloader, optimizer, margin)
        avg_loss = sum(losses)/len(losses)
        dist_a_p = sum(dist_a_p)/len(dist_a_p)
        dist_a_n = sum(dist_a_n)/len(dist_a_n)
        lengths = sum(lengths)/len(lengths)
        #dist_a_n = avg_loss - 0.2 - dist_a_p    # loss = margin + a_p + a_n => a_n = loss - margin - ap
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}, avg_desc_length: {lengths:.4f}")
        #test_acc = test(test_loader)
        #print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
        z = 0.00001
        if dist_a_p < z and dist_a_n < z and margin - 10*z < avg_loss < margin + 10*z:
            print("Stopping due to collapse of descriptors")
            import sys
            sys.exit(-1)




import torch_geometric.data.batch as geometric_batch
def train3(epoch, model, device):
    model.train()

    losses = []
    for batch in dataloader:
        # create single batch object
        datas = []
        for b in batch:
            datas += b.to_data_list()
        batch_all = geometric_batch.Batch.from_data_list(datas)

        batch_all = batch_all.to(device)
        optimizer.zero_grad()
        output = model(batch_all)

        loss = F.nll_loss(output, batch_all.id)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses.append(loss.item())

    return losses

if __name__ == '__main__2':
    from setup import Config
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cfg = Config(enable_writer=False)
    dataloader = DataLoader(dataset=cfg.DATASET, batch_size=15, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        losses = train3(epoch, model, device)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}")
        #test_acc = test(test_loader)
        #print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))