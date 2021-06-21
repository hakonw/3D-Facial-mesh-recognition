from os import makedirs, posix_fadvise
import os.path as osp
from random import Random

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, Linear as Lin, ReLU, BatchNorm1d as BN
# from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
#from torch_geometric.data import DataLoader  # Instead of this, use modified dataloader to not throw away data 
from meshfr.datasets.datasetGeneric import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

torch.manual_seed(1)
torch.cuda.manual_seed(1)

from tqdm import tqdm

# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

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

        self.sa1_module = SAModule(0.5, 0.2/2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4/2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)

    def forward(self, data):
        if isinstance(data, Batch):
            sa0_out = (data.x, data.pos, data.batch)
        elif isinstance(data, Data):
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
            sa0_out = (data.x, data.pos, batch)
        else:
            raise RuntimeError(f"Illegal data of type: {type(data)}")

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        #sa4_out = self.sa4_module(*sa3_out)
        
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        # x = F.relu(self.lin3(x))
        # x = F.normalize(x, dim=-1, p=2)  # L2 Normalization tips
        return x
        # return F.log_softmax(x, dim=-1)  # remember correct out shape

# Test 1

def train(epoch, dataloader, optimizer):
    model = None 
    train_loader = None 
    device = None
    model.train()

    losses = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            losses.append(loss.item())
    return losses

def test(loader):
    model = None
    device = None
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# def test_1_regular_poitnet():
#     path = osp.join(
#         osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
#     pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
#     train_dataset = ModelNet(path, '10', True, transform, pre_transform)
#     test_dataset = ModelNet(path, '10', False, transform, pre_transform)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
#                               num_workers=6)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
#                              num_workers=6)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Net().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(1, 201):
#         dataloader = None
#         losses = train(epoch, dataloader, optimizer)
#         avg_loss = sum(losses)/len(losses)
#         test_acc = test(test_loader)
#         print('Epoch: {:03d}, Test: {:.4f}, Avg-loss {:.4f}'.format(epoch, test_acc, avg_loss))


# Test 2 - poitnet++ with triplet loss
import torch_geometric.data.batch as geometric_batch
import meshfr.tripletloss.onlineTripletLoss as onlineTripletLoss
import meshfr.evaluation.metrics as metrics
def train2(epoch, model, device, dataloader, optimizer, margin, criterion):
    model.train()

    losses = []
    dist_a_p = []
    dist_a_n = []
    lengths = []

    max_losses = []
    max_dist_a_ps = [] 
    min_dist_a_ns = []
    for batch in dataloader:
        # create single batch object
        datas = []
        for b in batch:
            datas += b.to_data_list()
        batch_all = geometric_batch.Batch.from_data_list(datas)

        batch_all = batch_all.to(device)
        optimizer.zero_grad()
        descritors = model(batch_all)

        # Create dict again
        dic_descriptors = {}
        for i in range(len(batch_all.id)):
            id = batch_all.id[i].item()
            if id in dic_descriptors:
                dic_descriptors[id].append(descritors[i])
            else:
                dic_descriptors[id] = [descritors[i]]
        descritors = list(dic_descriptors.values())

        if True: 
            all = []
            labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
            for ident, listt in enumerate(descritors):
                all += listt
                labels += [ident] * len(listt)
            
            all = torch.stack(all).to("cpu")
            labels = torch.tensor(labels).to("cpu")
            
            # loss, fraction_positive_triplets = onlineTripletLoss.batch_all_triplet_loss(labels=labels, embeddings=all, margin=margin)
            # # print(fraction_positive_triplets)
            loss, max_loss, max_dist_a_p, min_dist_a_n = onlineTripletLoss.batch_hard_triplet_loss(labels=labels, embeddings=all, margin=margin)

        if False:
            # Første id: anchor og pos
            # Alle andre: Negative
            negs = []
            for i in range(1, len(descritors)):
                for m in descritors[i]:
                    negs.append(m)
            anc = descritors[0][1].unsqueeze(0).expand(len(negs), -1)
            pos = descritors[0][0].unsqueeze(0).expand(len(negs), -1)
            negs = torch.stack(negs)
            loss = criterion(anc, pos, negs)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            losses.append(loss.item())
            dist_a_p.append(torch.dist(descritors[0][0].to("cpu"), descritors[0][1].to("cpu"), p=2).item())
            dist_a_n.append(torch.dist(descritors[1][0].to("cpu"), descritors[0][0].to("cpu"), p=2).item())
            lengths.append(torch.norm(descritors[0][0], 2))

            max_losses.append(max_loss.item())
            max_dist_a_ps.append(max_dist_a_p.item())
            min_dist_a_ns.append(min_dist_a_n.item())
    return losses, dist_a_p, dist_a_n, lengths, max_losses, max_dist_a_ps, min_dist_a_ns
    # return losses, dist_a_p, dist_a_n
    
def test_2_pointnet_triplet_loss():
    POST_TRANSFORM = T.Compose([T.NormalizeScale(), T.SamplePoints(num=1024)])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    
    start_epoch = 1  # re-written if starting from a loaded save

    DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    # Load dataset and split into train/test 
    dataset = datasetBU3DFE.BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)
    train_set, test_set = torch.utils.data.random_split(dataset, [80, 20])

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    
    dataloader_val = DataLoader(dataset=train_set, batch_size=5, shuffle=False, num_workers=0, drop_last=False)
    dataloader_test = DataLoader(dataset=test_set, batch_size=5, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    # print("Loading save")
    # model.load_state_dict(torch.load("./Pointnet-triplet-128desc-hard-1500.pt"))
    # start_epoch += 1500

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # loss = ||ap|| - ||an|| + margin.  neg loss => ||an|| >>> ||ap||, at least margin over

    margin = 0.2

    # criterion for the naive approach
    criterion = torch.nn.TripletMarginLoss(margin=margin)

    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(start_epoch, 601):
        losses, dist_a_p, dist_a_n, lengths, max_losses, max_dist_a_ps, min_dist_a_ns = train2(epoch, model, device, dataloader, optimizer, margin, criterion)
        # losses, dist_a_p, dist_a_n = train2(epoch, model, device, dataloader, optimizer, margin, criterion)
        avg_loss = sum(losses)/len(losses)
        dist_a_p = sum(dist_a_p)/len(dist_a_p)
        dist_a_n = sum(dist_a_n)/len(dist_a_n)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}, avg_desc_length: {(sum(lengths)/len(lengths)):.2f}, max_loss: {max(max_losses):.4f}, max_dist_a_p: {max(max_dist_a_ps):.4f}, min_dist_a_n: {min(min_dist_a_ns):.4f}")
        # print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}")
        z = 0.00001
        if dist_a_p < z and dist_a_n < z and margin - 10*z < avg_loss < margin + 10*z:
            print("Stopping due to collapse of descriptors"); import sys; sys.exit(-1)

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                # descriptor_dict = metrics.data_dict_to_descriptor_dict(model=model, device=device, data_dict=cfg.DATASET_HELPER.get_cached_dataset(), desc="Evaluation/Test", leave_tqdm=False)
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_test, device=device)
                print("RANK-1-testdata", metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict).__str_short__())
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_val, device=device)
                print("RANK-1-traindata", metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict).__str_short__())

        # if epoch % 100 == 0:
        #     name = f"./Pointnet-triplet-128desc-hard-{epoch}.pt"
        #     print(f"Saving {name}")
        #     torch.save(model.state_dict(), name)

# Test 3 - poitnet++ with softmax
import torch_geometric.data.batch as geometric_batch
def train3(epoch, model, device, dataloader, optimizer):
    model.train()

    losses = []
    total_samples = 0
    total_correct_samples = 0
    for batch in dataloader:
        # # create single batch object
        # datas = []
        # for b in batch:
        #     datas += b.to_data_list()
        #     #  print(b.id) #  tensor([22, 85, 84, 33, 80, 26, 71, 97, 62,  6, 34, 73,  2, 18, 29])
        # batch_all = geometric_batch.Batch.from_data_list(datas)

        for batch_all in batch:
            batch_all = batch_all.to(device)
            optimizer.zero_grad()
            output = model(batch_all)
            # print(output.max(1))
            # print(batch_all.id)
            loss = F.nll_loss(output, batch_all.id)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_samples += batch_all.id.shape[0]
                correct_samples = output.max(1)[1].eq(batch_all.id).sum().item()
                total_correct_samples += correct_samples
                print(correct_samples, end=" ")
                losses.append(loss.item())


    print(f"\taka {total_correct_samples}/{total_samples} ({total_correct_samples/total_samples:0.3f}),", end=" ")
    return losses

def test_3_poitnet_softmax():
    from setup import Config
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cfg = Config(enable_writer=False)
    dataloader = DataLoader(dataset=cfg.DATASET, batch_size=50, shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(dataset=cfg.DATASET, batch_size=20, shuffle=True, num_workers=0, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(1, 601):
        losses = train3(epoch, model, device, dataloader, optimizer)
        avg_loss = sum(losses)/len(losses)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}")
        #test_acc = test(test_loader)

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()

                correct = 0
                length = 0
                for batch in dataloader_test:
                    datas = []
                    for b in batch:
                        datas += b.to_data_list()
                    batch_all = geometric_batch.Batch.from_data_list(datas).to(device)
                    length += batch_all.id.shape[0]
                    output = model(batch_all)
                    pred = output.max(1)[1]  # Values, indicies
                    correct += pred.eq(batch_all.id).sum().item()
                print(f"Evaluation Accuracy: {correct / length:4f}")

# Test 4 - gcnconv
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, BatchNorm, TopKPooling
class TestNet55(torch.nn.Module):
    def __init__(self):
        super(TestNet55, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.activation = ReLU()

        self.conv1 = GCNConv(in_channels=3, out_channels=64)
        self.batch1 = BatchNorm(in_channels=self.conv1.out_channels)

        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.batch2 = BatchNorm(in_channels=self.conv2.out_channels)

        self.conv3 = GCNConv(in_channels=128, out_channels=128)
        self.batch3 = BatchNorm(in_channels=self.conv3.out_channels)

        self.conv4 = GCNConv(in_channels=128, out_channels=256)
        self.batch4 = BatchNorm(in_channels=self.conv4.out_channels)

        self.conv5 = GCNConv(in_channels=256, out_channels=512)
        self.batch5 = BatchNorm(in_channels=self.conv5.out_channels)

        self.pooling1 = TopKPooling(in_channels=1, ratio=1024)

        self.fc1 = Linear(512, 256)
        self.fc2 = Linear(256, 256)
        self.fc3 = Linear(256, 100)


    def forward(self, data):
        pos, edge_index, batch = data.pos, data.edge_index, data.batch
        x = pos

        x, edge_index, edge_attr, batch, perm, score = self.pooling1(x=x, edge_index=edge_index, batch=batch)

        x = self.conv1(x, edge_index)
        x = self.batch1(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.batch2(x)
        x = self.activation(x)

        x = self.conv3(x, edge_index)
        x = self.batch3(x)
        x = self.activation(x)

        x = self.conv4(x, edge_index)
        x = self.batch4(x)
        x = self.activation(x)

        x = self.conv5(x, edge_index)
        x = self.batch5(x)
        x = self.activation(x)
                
        x = scatter(x, batch, dim=0)  # Unsure if the correct
        # return scatter(x, batch, dim=0, dim_size=x.shape[0], reduce='add') ?

        x = self.activation(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

import torch_geometric.data.batch as geometric_batch
def train55(epoch, model, device, dataloader, optimizer):
    model.train()

    losses = []
    total_samples = 0
    total_correct_samples = 0
    for batch in dataloader:
        # create single batch object
        # datas = []
        #for b in batch:
        #    datas += b.to_data_list()
        #batch_all = geometric_batch.Batch.from_data_list(datas)

        # due to memory restraints, do a mini batch with each type of face (should be randomized tho)
        # 50 folk (dataloader size) per gang, 25 ansikt per person, aka 100/50 * 25 loops
        for batch_all in batch:
            batch_all = batch_all.to(device)
            optimizer.zero_grad()
            output = model(batch_all)
            loss = F.nll_loss(output, batch_all.id)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_samples += batch_all.id.shape[0]
                correct_samples = output.max(1)[1].eq(batch_all.id).sum().item()
                total_correct_samples += correct_samples
                print(correct_samples, end=" ")
                losses.append(loss.item())


    print(f"\taka {total_correct_samples}/{total_samples} ({total_correct_samples/total_samples:0.3f}),", end=" ")
    return losses

def test_4_convnet_softmax():
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True),T.NormalizeScale()])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    
    start_epoch = 1

    DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    dataset = datasetBU3DFE.BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader = DataLoader(dataset=dataset, batch_size=50, shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(dataset=dataset, batch_size=25, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestNet55().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(1, 601):
        losses = train55(epoch, model, device, dataloader, optimizer)
        avg_loss = sum(losses)/len(losses)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()

                correct = 0
                length = 0
                for batch in dataloader_test:
                    for b in batch:
                        b = b.to(device)
                        length += b.id.shape[0]
                        pred = model(b).max(1)[1]  # Values, indicies fra max
                        correct += pred.eq(b.id).sum().item()
                print(f"Evaluation Accuracy: {(correct / length):6f} ({correct}/{length})")


class TestNet55_desc(torch.nn.Module):
    def __init__(self):
        super(TestNet55_desc, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.activation = ReLU()    

        # org: 3,64,128,128,256,512,pool,512,256,256,128
        self.conv1 = GCNConv(in_channels=3, out_channels=64)
        self.batch1 = BatchNorm(in_channels=self.conv1.out_channels)

        self.conv2 = GCNConv(in_channels=64, out_channels=94)
        self.batch2 = BatchNorm(in_channels=self.conv2.out_channels)

        self.conv3 = GCNConv(in_channels=94, out_channels=256)
        self.batch3 = BatchNorm(in_channels=self.conv3.out_channels)

        # self.conv4 = GCNConv(in_channels=256, out_channels=512)
        # self.batch4 = BatchNorm(in_channels=self.conv4.out_channels)

        # self.conv5 = GCNConv(in_channels=256, out_channels=512)
        # self.batch5 = BatchNorm(in_channels=self.conv5.out_channels)

        # self.pooling1 = TopKPooling(in_channels=1, ratio=1024)

        self.fc0 = Linear(256, 128)

        self.fc1 = Linear(128, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 100)


    def forward(self, data):
        if isinstance(data, Batch):  # Batch before data, as a batch is a data 
            pos, edge_index, batch = data.pos, data.edge_index, data.batch
        elif isinstance(data, Data):
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
            pos, edge_index, batch = data.pos, data.edge_index, batch
        else:
            raise RuntimeError(f"Illegal data of type: {type(data)}")
        x = pos

        # x, edge_index, edge_attr, batch, perm, score = self.pooling1(x=x, edge_index=edge_index, batch=batch)

        x = self.conv1(x, edge_index)
        x = self.batch1(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.batch2(x)
        x = self.activation(x)

        x = self.conv3(x, edge_index)
        x = self.batch3(x)
        x = self.activation(x)

        x = self.fc0(x)
        x = self.activation(x)

        # x = self.conv4(x, edge_index)
        # x = self.batch4(x)
        # x = self.activation(x)



        # x = self.conv5(x, edge_index)
        # x = self.batch5(x)
        # x = self.activation(x)
        # x = torch.stack(torch.split(x, batch.size(0)//data.num_graphs, 0))
        # x = torch.flatten(x, 1, 2)

        x = scatter(x, batch, dim=0) # , reduce="mean")  # Unsure if the correct
        # print(scatter(x, batch, dim=0, dim_size=x.shape[0], reduce='add'))
        # x = global_max_pool(x, batch)

        x = self.activation(self.fc1(x))
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        # x = F.normalize(x, dim=-1, p=2)  # L2 Normalization tips
        return x
        # return F.log_softmax(x, dim=-1)


class TestNet55_descv2(torch.nn.Module):
    def __init__(self):
        super(TestNet55_descv2, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.activation = ReLU()

        # org: 3,64,128,128,256,512,pool,512,256,256,128
        self.conv1 = GCNConv(in_channels=3, out_channels=16)
        self.batch1 = BatchNorm(in_channels=self.conv1.out_channels)

        self.conv11 = GCNConv(in_channels=16, out_channels=32)
        self.batch11 = BatchNorm(in_channels=self.conv11.out_channels)

        self.conv12 = GCNConv(in_channels=32, out_channels=64)
        self.batch12 = BatchNorm(in_channels=self.conv12.out_channels)

        self.conv2 = GCNConv(in_channels=64, out_channels=94)
        self.batch2 = BatchNorm(in_channels=self.conv2.out_channels)

        self.conv3 = GCNConv(in_channels=94, out_channels=256)
        self.batch3 = BatchNorm(in_channels=self.conv3.out_channels)

        # self.pooling1 = TopKPooling(in_channels=1, ratio=512)

        self.fc0 = Linear(256, 128)

        self.fc1 = Linear(128, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 128)

        # self.fc1 = Linear(256, 256)
        # self.fc2 = Linear(256, 256)
        # self.fc3 = Linear(256, 256)


    def forward(self, data):
        if isinstance(data, Batch):  # Batch before data, as a batch is a data
            pos, edge_index, batch = data.pos, data.edge_index, data.batch
        elif isinstance(data, Data):
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
            pos, edge_index, batch = data.pos, data.edge_index, batch
        elif isinstance(data, tuple):
            pos, edge_index = data
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        else:
            raise RuntimeError(f"Illegal data of type: {type(data)}")
        x = pos

        x = self.activation(self.batch1(self.conv1(x, edge_index)))
        x = self.activation(self.batch11(self.conv11(x, edge_index)))
        x = self.activation(self.batch12(self.conv12(x, edge_index)))

        x = self.activation(self.batch2(self.conv2(x, edge_index)))
        x = self.activation(self.batch3(self.conv3(x, edge_index)))

        x = self.activation(self.fc0(x))  # Per node fc
        
        # x = scatter(x, batch, dim=0, reduce="max") # , reduce="mean")  # Unsure if the correct
        # print(scatter(x, batch, dim=0, dim_size=x.shape[0], reduce='add'))
        x = global_max_pool(x, batch)


        # preshape = x.shape[0]  # torch.Size([476160, 128])
        # x = torch.stack(torch.split(x, batch.size(0)//data.num_graphs, 0))
        # assert preshape == x.shape[0] * x.shape[1] # torch.Size([465, 1024, 128])

        # D: Not
        # x = torch.flatten(x, 1, 2)  # torch.Size([465, 131072])

        # A: Maxpool per node
        # print(torch.max(x, dim=2)[0].shape) torch.Size([450, 1024])
        # x = torch.max(x, dim=2)[0]

        # B: Maxpool per filter
        # print(torch.max(x, dim=1)[0].shape) torch.Size([450, 128])
        # x = torch.max(x, dim=1)[0]

        # C: Pool it somehow
        # x, edge_index, edge_attr, batch, perm, score = self.pooling1(x=x, edge_index=edge_index, batch=batch)
        # print(x.shape)
        # x = self.activation(self.batch3(self.conv4(x, edge_index)))
        # x = torch.stack(torch.split(x, self.pooling1.ratio, 0))
        # x = torch.flatten(x, 1, 2)

        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# Test 5 - convnet with triplet loss
import torch_geometric.data.batch as geometric_batch
import meshfr.evaluation.metrics as metrics
def train5(epoch, model, device, dataloader, optimizer, margin, criterion):
    model.train()

    losses = []
    dist_a_p = []
    dist_a_n = []
    lengths = []

    max_losses = []
    max_dist_a_ps = [] 
    min_dist_a_ns = []
    for batch in dataloader:
        # create single batch object
        datas = []
        for b in batch:
            datas += b.to_data_list()
        batch_all = geometric_batch.Batch.from_data_list(datas)

        batch_all = batch_all.to(device)
        optimizer.zero_grad()
        descritors = model(batch_all)

        # Create dict again
        dic_descriptors = {}
        for i in range(len(batch_all.id)):
            id = batch_all.id[i].item()
            if id in dic_descriptors:
                dic_descriptors[id].append(descritors[i])
            else:
                dic_descriptors[id] = [descritors[i]]
        descritors = list(dic_descriptors.values())

        if True:
            all = []
            labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
            for ident, listt in enumerate(descritors):
                all += listt
                labels += [ident] * len(listt)
            
            all = torch.stack(all).to("cpu")
            labels = torch.tensor(labels).to("cpu")
            
            # loss, fraction_positive_triplets = onlineTripletLoss.batch_all_triplet_loss(labels=labels, embeddings=all, margin=margin)
            # print(fraction_positive_triplets)
            loss, max_loss, max_dist_a_p, min_dist_a_n = onlineTripletLoss.batch_hard_triplet_loss(labels=labels, embeddings=all, margin=margin)

        if False:
            # Første id: anchor og pos
            # Alle andre: Negative
            negs = []
            for i in range(1, len(descritors)):
                for m in descritors[i]:
                    negs.append(m)
            anc = descritors[0][1].unsqueeze(0).expand(len(negs), -1)
            pos = descritors[0][0].unsqueeze(0).expand(len(negs), -1)
            negs = torch.stack(negs)
            loss = criterion(anc, pos, negs)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            losses.append(loss.item())
            dist_a_p.append(0)
            dist_a_n.append(0)
            # dist_a_p.append(torch.dist(descritors[0][0].to("cpu"), descritors[0][1].to("cpu"), p=2).item())
            # dist_a_n.append(torch.dist(descritors[1][0].to("cpu"), descritors[0][0].to("cpu"), p=2).item())
            
            lengths.append(torch.norm(descritors[0][0], 2))
            try: 
                max_losses.append(max_loss.item())
                max_dist_a_ps.append(max_dist_a_p.item())
                min_dist_a_ns.append(min_dist_a_n.item())
            except:
                pass
    if len(max_losses) == 0:
        return losses, dist_a_p, dist_a_n
    return losses, dist_a_p, dist_a_n, lengths, max_losses, max_dist_a_ps, min_dist_a_ns

import meshfr.datasets.datasetBU3DFEv2 as datasetBU3DFE
import math
def test_5_convnet_triplet():
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    
    start_epoch = 1  # re-written if starting from a loaded save

    # DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    # BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    # dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    import pickle

    import meshfr.datasets.reduction_transform as reduction_transform
    # with torch.no_grad():
    #     pre_redux = reduction_transform.SimplifyQuadraticDecimationBruteForce(2048)
    #     from tqdm import tqdm
    #     for identity, faces in tqdm(dataset_cached.items()):
    #         for name, data in faces.items():
    #             dataset_cached[identity][name] = pre_redux(data)
    #     pickle.dump(dataset_cached, open("BU-3DFE_cache-reduced.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    dataset_cached = pickle.load(open("BU-3DFE_cache-reduced.p", "rb"))
    print("Saved/loaded data")

    # Load dataset and split into train/test 
    dataset_bu3dfe = datasetBU3DFE.BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)
    train_set, test_set = torch.utils.data.random_split(dataset_bu3dfe, [80, 20])

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader_bu3dfe = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    dataloader_bu3dfe_train = DataLoader(dataset=train_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bu3dfe_test = DataLoader(dataset=test_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bu3dfe_all = DataLoader(dataset=dataset_bu3dfe, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    #dataloader = dataloader_bu3dfe

    import meshfr.datasets.datasetBosphorus as datasetBosphorus
    from meshfr.datasets.datasetGeneric import GenericDataset
    bosphorus_path = "/lhome/haakowar/Downloads/Bosphorus/BosphorusDB"
    # bosphorus_dict = datasetBosphorus.get_bosphorus_dict("/tmp/invalid", pickled=True)
    bosphorus_dict = datasetBosphorus.get_bosphorus_dict(bosphorus_path, pickled=True, force=False, picke_name="/tmp/Bosphorus_cache-full-2pass.p")
    dataset_bosphorus = GenericDataset(bosphorus_dict, POST_TRANSFORM)
    bosphorus_train_set, bosphorus_test_set = torch.utils.data.random_split(dataset_bosphorus, [80, 25])
    dataloader_bosphorus_test = DataLoader(dataset=bosphorus_test_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bosphorus_train = DataLoader(dataset=bosphorus_train_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bosphorus_all = DataLoader(dataset=dataset_bosphorus, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    # dataloader = DataLoader(dataset=bosphorus_train_set, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    from meshfr.datasets.datasetFRGC import get_frgc_dict
    frgc_path = "/lhome/haakowar/Downloads/FRGCv2/Data/"
    dataset_frgc_fall_2003 = get_frgc_dict(frgc_path + "Fall2003range", pickled=True,force=False, picke_name="FRGCv2-fall2003_cache.p")
    dataset_frgc_spring_2003 = get_frgc_dict(frgc_path + "Spring2003range", pickled=True, force=False, picke_name="FRGCv2-spring2003_cache.p")
    dataset_frgc_spring_2004 = get_frgc_dict(frgc_path + "Spring2004range", pickled=True, force=False, picke_name="FRGCv2-spring2004_cache.p")
    dataset_frgc_fall_2003 = GenericDataset(dataset_frgc_fall_2003, POST_TRANSFORM)
    dataset_frgc_spring_2003 = GenericDataset(dataset_frgc_spring_2003, POST_TRANSFORM)
    dataset_frgc_spring_2004 = GenericDataset(dataset_frgc_spring_2004, POST_TRANSFORM)
    dataloader_frgc_test = DataLoader(dataset=dataset_frgc_fall_2003, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataset_frgc = torch.utils.data.ConcatDataset([dataset_frgc_spring_2003, dataset_frgc_spring_2004])
    dataloader_frgc_train = DataLoader(dataset=dataset_frgc, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader = DataLoader(dataset=dataset_frgc, batch_size=5, shuffle=True, num_workers=0, drop_last=True)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = TestNet55_desc().to(device)
    model = TestNet55_descv2().to(device)

    # print("Loading save")
    # model.load_state_dict(torch.load("./Testnet55_desc-triplet-128desc-500.pt"))
    # start_epoch += 500

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
    # loss = ||ap|| - ||an|| + margin.  neg loss => ||an|| >>> ||ap||, at least margin over

    margin = 0.2

    # critering used in naive approach
    criterion = torch.nn.TripletMarginLoss(margin=margin)  # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss   mean or sum reduction possible

    LOG = ask_for_writer(dataloader, optimizer)
    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(start_epoch, 405):
        losses, dist_a_p, dist_a_n, lengths, max_losses, max_dist_a_ps, min_dist_a_ns = train5(epoch, model, device, dataloader, optimizer, margin, criterion)
        # losses, dist_a_p, dist_a_n = train5(epoch, model, device, dataloader, optimizer, margin, criterion)
        avg_loss = sum(losses)/len(losses)
        dist_a_p = sum(dist_a_p)/len(dist_a_p)
        dist_a_n = sum(dist_a_n)/len(dist_a_n)
        # lengths = sum(lengths)/len(lengths)
        #dist_a_n = avg_loss - 0.2 - dist_a_p    # loss = margin + a_p + a_n => a_n = loss - margin - ap
        # print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}")
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}, avg_desc_length: {(sum(lengths)/len(lengths)):.2f}, max_loss: {max(max_losses):.4f}, max_dist_a_p: {max(max_dist_a_ps):.4f}, min_dist_a_n: {min(min_dist_a_ns):.4f}")
        z = 0.00001

        LOG.add_scalar("Loss/train-avg", avg_loss, epoch)
        LOG.add_scalar("Distance/anchor-positive", dist_a_p, epoch)
        LOG.add_scalar("Distance/anchor-negative", dist_a_n, epoch)
        try:
            LOG.add_scalar("Loss/train-max", max(max_losses), epoch)
            LOG.add_scalar("Distance/max-anchor-positive", max(max_dist_a_ps), epoch)
            LOG.add_scalar("Distance/min-anchor-negative", min(min_dist_a_ns), epoch)
            LOG.add_scalar("Distance/length", sum(lengths)/len(lengths), epoch)
        except:
            pass

        if dist_a_p < z and dist_a_n < z and margin - 10*z < avg_loss < margin + 10*z:
            print("Stopping due to collapse of descriptors"); import sys; sys.exit(-1)
        if math.isnan(avg_loss):
            print("Stopping due to nan"); import sys; sys.exit(-1)

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                # descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bu3dfe_test, device=device)
                # metric = metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
                # print("RANK-1-testdata (BU-3DFE)", metric.__str_short__())
                # for m in ["tp", "fp", "accuracy"]:
                #     LOG.add_scalar("metric-" + m + "/val", getattr(metric, m), epoch)
                
                # descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bu3dfe_train, device=device)
                # metric = metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
                # print("RANK-1-traindata (BU-3DFE)", metric.__str_short__())
                # for m in ["tp", "fp", "accuracy"]:
                #     LOG.add_scalar("metric-" + m + "/train", getattr(metric, m), epoch)
                
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bu3dfe_all, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
                print("RANK-1-all (BU-3DFE)", metric.__str_short__())
                for m in ["tp", "fp", "accuracy"]:
                    LOG.add_scalar("metric-bu3dfe-" + m + "/test", getattr(metric, m), epoch)

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                # bosphorus_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bosphorus_test, device=device)
                # metric = metrics.get_metric_gallery_set_vs_probe_set_bosphorus(bosphorus_descriptor_dict)
                # print("RANK-1-testdata (bosphorus)", metric.__str_short__())
                # for m in ["tp", "fp", "accuracy"]:
                #     LOG.add_scalar("metric-bosphorus-" + m + "/test", getattr(metric, m), epoch)

                # bosphorus_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bosphorus_train, device=device)
                # metric = metrics.get_metric_gallery_set_vs_probe_set_bosphorus(bosphorus_descriptor_dict)
                # print("RANK-1-traindata (bosphorus)", metric.__str_short__())
                # for m in ["tp", "fp", "accuracy"]:
                #     LOG.add_scalar("metric-bosphorus-" + m + "/train", getattr(metric, m), epoch)

                bosphorus_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bosphorus_all, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_bosphorus(bosphorus_descriptor_dict)
                print("RANK-1-all (bosphorus)", metric.__str_short__())
                for m in ["tp", "fp", "accuracy"]:
                    LOG.add_scalar("metric-bosphorus-" + m + "/test", getattr(metric, m), epoch)

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                frgc_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_frgc_test, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_frgc(frgc_descriptor_dict)
                print("RANK-1-test (FRGC)", metric.__str_short__())
                for m in ["tp", "fp", "accuracy"]:
                    LOG.add_scalar("metric-frgc-" + m + "/test", getattr(metric, m), epoch)
                    
                frgc_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_frgc_train, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_frgc(frgc_descriptor_dict)
                print("RANK-1-train (FRGC)", metric.__str_short__())
                for m in ["tp", "fp", "accuracy"]:
                    LOG.add_scalar("metric-frgc-" + m + "/train", getattr(metric, m), epoch)


        # if epoch % 100 == 0:
        #     name = f"./Testnet55_desc-triplet-128desc-8020-{epoch}.pt"
        #     print(f"Saving {name}")
        #     torch.save(model.state_dict(), name)


# Test 6, training with softmax, checking with embeddings
class NetPointnetDuo(torch.nn.Module):
    def __init__(self):
        super(NetPointnetDuo, self).__init__()
        torch.manual_seed(1)

        self.sa1_module = SAModule(0.5, 0.2/1, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4/1, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)  # Embeddings
        self.lin4 = Lin(128, 105)  # Softmax, 100 possible different classes, but not all are used, is that bad?

        self.embeddings = False

    def forward(self, data):
        if isinstance(data, Batch):
            sa0_out = (data.x, data.pos, data.batch)
        elif isinstance(data, Data):
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
            sa0_out = (data.x, data.pos, batch)
        else:
            raise RuntimeError(f"Illegal data of type: {type(data)}")

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        if self.embeddings:
            return x
        else:
            x = F.relu(x)
            x = self.lin4(x)
            return F.log_softmax(x, dim=-1)


class TestNet55_desc_softmax(torch.nn.Module):
    def __init__(self):
        super(TestNet55_desc_softmax, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.embeddings = False

        self.activation = ReLU()

        # org: 3,64,128,128,256,512,pool,512,256,256,128
        self.conv1 = GCNConv(in_channels=3, out_channels=64)
        self.batch1 = BatchNorm(in_channels=self.conv1.out_channels)

        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.batch2 = BatchNorm(in_channels=self.conv2.out_channels)

        self.conv3 = GCNConv(in_channels=128, out_channels=256)
        self.batch3 = BatchNorm(in_channels=self.conv3.out_channels)

        self.fc0 = Linear(256, 256)

        self.fc1 = Linear(256, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 105)

    def forward(self, data):
        if isinstance(data, Batch):  # Batch before data, as a batch is a data
            pos, edge_index, batch = data.pos, data.edge_index, data.batch
        elif isinstance(data, Data):
            batch = torch.zeros(data.pos.size(
                0), dtype=torch.long, device=data.pos.device)
            pos, edge_index, batch = data.pos, data.edge_index, batch
        else:
            raise RuntimeError(f"Illegal data of type: {type(data)}")
        x = pos

        x = self.conv1(x, edge_index)
        x = self.batch1(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.batch2(x)
        x = self.activation(x)

        x = self.conv3(x, edge_index)
        x = self.batch3(x)
        x = self.activation(x)

        x = self.fc0(x)
        x = self.activation(x)

        x = scatter(x, batch, dim=0)

        x = self.activation(self.fc1(x))
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        if self.embeddings:
            return x
        else:
            x = self.activation(x)
            x = self.fc3(x)
            return F.log_softmax(x, dim=-1)


import torch_geometric.data.batch as geometric_batch
def train6(epoch, model, device, dataloader, optimizer):
    model.train()
    model.embeddings = False

    losses = []
    total_samples = 0
    total_correct_samples = 0
    for major_batch in dataloader:
        for minor_batch in major_batch:
            minor_batch = minor_batch.to(device)
            optimizer.zero_grad()
            output = model(minor_batch)
            loss = F.nll_loss(output, minor_batch.id)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_samples += minor_batch.id.shape[0]
                correct_samples = output.max(1)[1].eq(minor_batch.id).sum().item()
                total_correct_samples += correct_samples
                print(correct_samples, end=" ")
                losses.append(loss.item())

    print(f"\taka {total_correct_samples}/{total_samples} ({total_correct_samples/total_samples:0.3f}),", end=" ")
    return losses


def test_6_softmax_embeddings():
    POST_TRANSFORM = T.Compose([T.NormalizeScale(), T.SamplePoints(num=1024)])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    
    start_epoch = 1  # re-written if starting from a loaded save

    DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    # Load dataset and split into train/test 
    dataset = datasetBU3DFE.BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)
    train_set, test_set = torch.utils.data.random_split(dataset, [80, 20])

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader = DataLoader(dataset=train_set, batch_size=40, shuffle=True, num_workers=0, drop_last=True)
    
    dataloader_val = DataLoader(dataset=train_set, batch_size=40, shuffle=False, num_workers=0, drop_last=False)
    dataloader_test = DataLoader(dataset=test_set, batch_size=20, shuffle=False, num_workers=0, drop_last=False)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NetPointnetDuo().to(device)

    print("Loading save")
    model.load_state_dict(torch.load("./test6-softmax-1100.pt"))
    start_epoch += 1100

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)

    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(start_epoch, 1501):
        losses = train6(epoch, model, device, dataloader, optimizer)
        avg_loss = sum(losses)/len(losses)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                model.embeddings = False

                length = 0
                correct = 0
                for major_batch in dataloader_val:
                    for minor_batch in major_batch: 
                        output = model(minor_batch.to(device))
                        length += minor_batch.id.shape[0]
                        correct += output.max(1)[1].eq(minor_batch.id).sum().item()
                        # sum of (maximum ident from output equals batch_all)

                print(f"Evaluation Accuracy: {(correct / length):6f} ({correct}/{length})")

                model.embeddings = True
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_test, device=device)
                print("RANK-1-testdata", metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict).__str_short__())
                
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_val, device=device)
                print("RANK-1-traindata", metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict).__str_short__())

        if epoch % 100 == 0:
            name = f"./test6-softmax-{epoch}.pt"
            print(f"Saving {name}")
            torch.save(model.state_dict(), name)

def test_7_softmax_embeddings2():
    # def sampling(s): return read_bnt.data_simple_sample(s, 2048*2)
    # POST_TRANSFORM_sample = T.Compose([sampling, T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    # POST_TRANSFORM = T.Compose([T.NormalizeScale(), T.SamplePoints(num=1024)])
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    start_epoch = 1  # re-written if starting from a loaded save

    # DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    # BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    # dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    import pickle
    import meshfr.datasets.reduction_transform as reduction_transform
    # with torch.no_grad():
    #     pre_redux = reduction_transform.SimplifyQuadraticDecimationBruteForce(2048)
    #     from tqdm import tqdm
    #     for identity, faces in tqdm(dataset_cached.items()):
    #         for name, data in faces.items():
    #             dataset_cached[identity][name] = pre_redux(data)
    #     pickle.dump(dataset_cached, open("BU-3DFE_cache-reduced.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    dataset_cached = pickle.load(open("BU-3DFE_cache-reduced.p", "rb"))
    print("Saved/loaded data")

    # Load dataset and split into train/test
    dataset = datasetBU3DFE.BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)
    train_set, test_set = torch.utils.data.random_split(dataset, [80, 20])

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader_bu3dfe = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    dataloader_bu3dfe_train = DataLoader(dataset=train_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bu3dfe_test = DataLoader(dataset=test_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bu3dfe_all = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    #dataloader = dataloader_bu3dfe

    import meshfr.datasets.datasetBosphorus as datasetBosphorus
    from meshfr.datasets.datasetGeneric import GenericDataset
    bosphorus_path = "/lhome/haakowar/Downloads/Bosphorus/BosphorusDB"
    # bosphorus_dict = datasetBosphorus.get_bosphorus_dict("/tmp/invalid", pickled=True)
    bosphorus_dict = datasetBosphorus.get_bosphorus_dict(bosphorus_path, pickled=False, force=False, picke_name="/tmp/Bosphorus_cache-full-2pass-1000.p")
    dataset_bosphorus = GenericDataset(bosphorus_dict, POST_TRANSFORM)
    bosphorus_train_set, bosphorus_test_set = torch.utils.data.random_split(dataset_bosphorus, [80, 25])
    dataloader_bosphorus_test = DataLoader(dataset=bosphorus_test_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_bosphorus_train = DataLoader(dataset=bosphorus_train_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader = DataLoader(dataset=bosphorus_train_set,batch_size=20, shuffle=True, num_workers=4, drop_last=True)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model = NetPointnetDuo().to(device)
    model = model = TestNet55_desc_softmax().to(device)

    # print("Loading save")
    # model.load_state_dict(torch.load("./Testnet55_desc-triplet-128desc-500.pt"))
    # start_epoch += 500

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    LOG = ask_for_writer(dataloader, optimizer)
    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(start_epoch, 405):
        losses = train6(epoch, model, device, dataloader, optimizer)
        avg_loss = sum(losses)/len(losses)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}")


        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                model.embeddings = False

                length = 0
                correct = 0
                for major_batch in dataloader_bosphorus_train:
                    for minor_batch in major_batch: 
                        output = model(minor_batch.to(device))
                        length += minor_batch.id.shape[0]
                        correct += output.max(1)[1].eq(minor_batch.id).sum().item()
                print(f"Evaluation Accuracy (train): {(correct / length):6f} ({correct}/{length})")

                model.embeddings = True

                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bu3dfe_all, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
                print("RANK-1-all (BU-3DFE)", metric.__str_short__())

                bosphorus_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bosphorus_test, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_bosphorus(bosphorus_descriptor_dict)
                print("RANK-1-testdata (bosphorus)", metric.__str_short__())

                bosphorus_descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_bosphorus_train, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_bosphorus(bosphorus_descriptor_dict)
                print("RANK-1-traindata (bosphorus)", metric.__str_short__())

        # if epoch % 100 == 0:
        #     name = f"./Testnet55_desc-triplet-128desc-8020-{epoch}.pt"
        #     print(f"Saving {name}")
        #     torch.save(model.state_dict(), name)


# Linear based siamese wiith binary classification
class Siamese_part(torch.nn.Module):
    def __init__(self):
        super(Siamese_part, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        # self.fc0 = Linear(128, 128)
        self.fc1 = Linear(256, 64)
        # self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)  # Combine both descritors

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # sigmoid to return <0, 1>
        assert x.shape[-1] == 1
        x = torch.squeeze(x)  # Make it [N] instead of [N, 1]
        return x


# Test 8 - convnet with siamese
def train8_sia(model, siam, device, dataloader, optimizer, criterion):
    model.train()

    losses = []
    correct = []
    correct_pos_pair = []
    correct_neg_pair = []
    for batch in dataloader:
        # create single batch object
        datas = []
        for b in batch:
            datas += b.to_data_list()
        batch_all = geometric_batch.Batch.from_data_list(datas)

        batch_all = batch_all.to(device)
        optimizer.zero_grad()
        descritors = model(batch_all)

        # Create dict again
        dic_descriptors = {}
        for i in range(len(batch_all.id)):
            # id = batch_all.id[i].item()
            id = batch_all.dataset_id[i]  # Use string name instead of idx id
            if id in dic_descriptors:
                dic_descriptors[id].append(descritors[i])
            else:
                dic_descriptors[id] = [descritors[i]]
        descritors = list(dic_descriptors.values())

        all = []
        labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
        for ident, listt in enumerate(descritors):
            all += listt
            labels += [ident] * len(listt)

        all = torch.stack(all).to(device)
        labels = torch.tensor(labels).to(device)

        # Create a NxN matrix of all vs all

        # print("all", all.shape)
        # print("labels", labels.shape)
        labels_combi = torch.combinations(labels, r=2, with_replacement=True)
        labels_combi = labels_combi[:, 0].eq(labels_combi[:, 1]).float()
        
        indecies = torch.arange(all.shape[0]).to(device); # print(indecies)
        indecies = torch.combinations(indecies, r=2, with_replacement=True)
        assert indecies.shape[0] == labels_combi.shape[0]
        # indiecies_org = indecies
        
        # balance
        # Find indecies for all positive.
        indecies_pos = indecies[labels_combi.eq(1)]
        label_pos = labels_combi[labels_combi.eq(1)]

        # Find indecies for alle negative pairs
        mask_neg = torch.randperm(indecies[labels_combi.eq(0)].shape[0])[:indecies_pos.shape[0]]
        indecies_neg = indecies[labels_combi.eq(0)][mask_neg]
        label_neg = labels_combi[labels_combi.eq(0)][mask_neg]
        # print("labels", labels_combi.shape)
        # print("sum labels", labels_combi.eq(1).sum().item())
        # print("indi", indecies.shape)
        # print("pos", indecies_pos.shape)
        # print("pos-test", indecies_pos[:, 0].eq(indecies_pos[:, 1]).sum())
        # print("neg", indecies_neg.shape)
        
        # Cobmine for all indecies to give to the model
        indecies = torch.cat((indecies_pos, indecies_neg), dim=0)
        labels_combi = torch.cat((label_pos, label_neg), dim=0)
        # print(labels_combi[:2600].sum())
        # print(labels_combi.sum())
        # print("cat", indecies.shape)
        
        middle = indecies_pos.shape[0]  # middle = labels_combi.shape[0]//2 becaome wrong when it was odd

        assert labels_combi[:middle].sum().eq(middle).item()  # The first half should all be "1"
        assert labels_combi[middle:].sum().eq(0).item()  # The seconds half should all be "0"

        descriptors_combi_1 = all[indecies[:, 0]]
        descriptors_combi_2 = all[indecies[:, 1]]
        # print("desc1", descriptors_combi_1.shape)
        # print("desc2", descriptors_combi_2.shape)
        
        result = siam(descriptors_combi_1, descriptors_combi_2)
        loss = criterion(result, labels_combi)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses.append(loss.item())
            correct.append(round((result>0.5).eq(labels_combi).sum().item()/labels_combi.shape[0], 4))
            results_pos_pair = result[labels_combi.eq(1)]
            results_neg_pair = result[labels_combi.eq(0)]
            correct_pos_pair.append((results_pos_pair>0.5).sum().item())
            correct_neg_pair.append((results_neg_pair<=0.5).sum().item())
    return losses, correct, correct_pos_pair, correct_neg_pair

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import meshfr.evaluation.realEvaluation as evaluation
import meshfr.datasets.reduction_transform as reduction_transform
from meshfr.datasets.datasetGeneric import ExtraTransform
import random 
import time 
import numpy as np
from datetime import timedelta
def test_8_convnet_triplet():
    # POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    # POST_TRANSFORM_Extra = T.Compose([])
    POST_TRANSFORM_Extra = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(5, axis=0),
        T.RandomRotate(5, axis=1),
        T.RandomRotate(5, axis=2)
        ])
    # POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.Center()])
    # POST_TRANSFORM = T.Compose([T.NormalizeScale(), T.RandomTranslate(0.01), T.RandomRotate(5, axis=0), T.RandomRotate(5, axis=1), T.RandomRotate(5, axis=2), T.SamplePoints(1024), reduction_transform.DelaunayIt(), T.FaceToEdge(remove_faces=True)])
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    start_epoch = 1  # re-written if starting from a loaded save
    default_epoch_per_log = 50

    
    valid = ["bu3dfe", "bosp", "frgc", "bosp+frgc", "bu3dfe+bosp+frgc"]
    train_on = "bosp"
    assert train_on in valid

    # Global properties
    pickled = True
    force = False
    sample = "bruteforce"  # 2pass, bruteforce, all or random
    # sample_size = [1024*8, 1024*12]
    sample_size = [1024*4, 1024*8]
    num_workers_train = 6

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # for the dataloader. As it is in memory, a high number is not needed, set to 0 if file desc errors https://pytorch.org/docs/stable/data.html
    # Alt,  check out   lsof | awk '{ print $2; }' | uniq -c | sort -rn | head
    # and               ulimit -n 4096
    def dload(dataset, batch_size, predicable, num_workers=0):
        if predicable:
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
        else:
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # Structure 
    #  Each dataset MAY use an in-memory-style dataset (the .p files)
    #  The dataset is split into:
    #    train:  Used strictly for training. Note that "_train" is used for evaluation on the dataset and "dataloader =" is used for training
    #    test:   Used for testing IF you want to check the same dataset OR as validation when testing on different datasets
    #    all:    The entire dataset. Used for testing on the entire dataset, and should not be used as training

    import meshfr.datasets.datasetBU3DFEv2 as datasetBU3DFEv2
    from meshfr.datasets.datasetGeneric import GenericDataset
    bu3dfe_path = "/lhome/haakowar/Downloads/BU_3DFE"
    bu3dfe_dict =  datasetBU3DFEv2.get_bu3dfe_dict(bu3dfe_path, pickled=pickled, force=force, picke_name="/tmp/Bu3dfe-2048.p", sample="bruteforce", sample_size=1024*2)
    dataset_bu3dfe = GenericDataset(bu3dfe_dict, POST_TRANSFORM)
    bu3dfe_train_set, bu3dfe_test_set = torch.utils.data.random_split(dataset_bu3dfe, [80, 20], generator=torch.Generator().manual_seed(42))
    bu3dfe_train_set = ExtraTransform(bu3dfe_train_set, POST_TRANSFORM_Extra)
    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    # TODO set back to 5, and 10
    dataloader_bu3dfe_train = dload(bu3dfe_train_set, batch_size=4, predicable=True)
    dataloader_bu3dfe_test  = dload(bu3dfe_test_set, batch_size=4, predicable=True)
    dataloader_bu3dfe_all   = dload(dataset_bu3dfe, batch_size=4, predicable=True)
    if train_on == "bu3dfe":
        dataloader = dload(bu3dfe_train_set, batch_size=10, predicable=False, num_workers=num_workers_train)


    import meshfr.datasets.datasetBosphorus as datasetBosphorus
    bosphorus_path = "/lhome/haakowar/Downloads/Bosphorus/BosphorusDB"
    bosphorus_dict = datasetBosphorus.get_bosphorus_dict(bosphorus_path, pickled=pickled, force=force, picke_name="/tmp/Bosphorus-2048-filter-new.p", sample=sample, sample_size=sample_size)
    dataset_bosphorus = GenericDataset(bosphorus_dict, POST_TRANSFORM)
    bosphorus_train_set, bosphorus_test_set = torch.utils.data.random_split(dataset_bosphorus, [80, 25], generator=torch.Generator().manual_seed(42))
    bosphorus_train_set = ExtraTransform(bosphorus_train_set, POST_TRANSFORM_Extra)
    dataloader_bosphorus_test  = dload(bosphorus_test_set, batch_size=4, predicable=True)
    dataloader_bosphorus_train = dload(bosphorus_train_set, batch_size=4, predicable=True)
    dataloader_bosphorus_all   = dload(dataset_bosphorus, batch_size=4, predicable=True)  # change to 2 for low memory
    if train_on == "bosp":
        dataloader = dload(bosphorus_train_set, batch_size=6, predicable=False, num_workers=num_workers_train)

    from meshfr.datasets.datasetFRGC import get_frgc_dict
    frgc_path = "/lhome/haakowar/Downloads/FRGCv2/Data/"
    dataset_frgc_fall_2003 = get_frgc_dict(frgc_path + "Fall2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-fall2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
    dataset_frgc_spring_2003 = get_frgc_dict(frgc_path + "Spring2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
    dataset_frgc_spring_2004 = get_frgc_dict(frgc_path + "Spring2004range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2004_cache-2048-new.p", sample=sample, sample_size=sample_size)
    dataset_frgc_fall_2003 = GenericDataset(dataset_frgc_fall_2003, POST_TRANSFORM)
    dataset_frgc_spring_2003 = GenericDataset(dataset_frgc_spring_2003, POST_TRANSFORM)
    dataset_frgc_spring_2004 = GenericDataset(dataset_frgc_spring_2004, POST_TRANSFORM)
    dataset_frgc_train = torch.utils.data.ConcatDataset([dataset_frgc_spring_2003])  # As per doc, Spring2003 is train, rest is val
    dataset_frgc_test = torch.utils.data.ConcatDataset([dataset_frgc_fall_2003, dataset_frgc_spring_2004])  # As per doc, Spring2003 is train, rest is val
    dataloader_frgc_train = dload(dataset_frgc_train, batch_size=2, predicable=True)
    dataloader_frgc_test = dload(dataset_frgc_test, batch_size=2, predicable=True)
    dataset_frgc_all = torch.utils.data.ConcatDataset([dataset_frgc_fall_2003, dataset_frgc_spring_2003, dataset_frgc_spring_2004])
    dataloader_frgc_all = dload(dataset_frgc_all, batch_size=5, predicable=True)
    if train_on == "frgc":
        dataloader = dload(dataset_frgc_train, batch_size=20, predicable=False, num_workers=num_workers_train)

    # from datasets.dataset3DFace import get_3dface_dict
    # d3face_path = "/lhome/haakowar/Downloads/3DFace_DB/3DFace_DB/"
    # d3face_dict = get_3dface_dict(d3face_path, pickled=pickled, force=force, picke_name="/tmp/3dface-12k.p", sample="bruteforce", sample_size=4096*12)
    # d3face_dataset_all = GenericDataset(d3face_dict, POST_TRANSFORM)
    # dataloader_3dface_all = dload(d3face_dataset_all, batch_size=5, predicable=True)
    # a = T.SamplePoints(1024)
    # print(a(d3face_dict["000"]["000_0"]))

    # Combination:
    # Trained on BOSP + FRGC, 
    if train_on == "bosp+frgc":
        dataset_frgc_bosp_train = torch.utils.data.ConcatDataset([bosphorus_train_set, dataset_frgc_train])
        dataloader = dload(dataset_frgc_bosp_train, batch_size=10, predicable=False, num_workers=num_workers_train)
        # TODO combine val data?
    
    if train_on == "bu3dfe+bosp+frgc":
        dataset_bu3dfe_frgc_bosp_train = torch.utils.data.ConcatDataset([bu3dfe_train_set, bosphorus_train_set, dataset_frgc_train])
        dataloader = dload(dataset_bu3dfe_frgc_bosp_train, batch_size=5, predicable=False, num_workers=num_workers_train)

    # Load the model
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print(f"PSA: using {device}")
    model = TestNet55_descv2().to(device)
    # model = Net().to(device)
    siam = Siamese_part().to(device)

    print("Loading save")
    model.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-17_lr1e-03_batchsize6_testing-bosp-norm-t001-r5-axis012/model-500.pt"))
    siam.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-17_lr1e-03_batchsize6_testing-bosp-norm-t001-r5-axis012/model-siam-500.pt"))
    start_epoch += 500
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()


    LOG = ask_for_writer(dataloader, optimizer)
    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    time_avg = 1
    for epoch in range(start_epoch, 6005):
        model.train()
        siam.train()
        start = time.time()
        losses, correct, correct_pos_pair, correct_neg_pair = train8_sia(model, siam, device, dataloader, optimizer, criterion)
        losses, correct, correct_pos_pair, correct_neg_pair = train8_sia(model, siam, device, dataloader, optimizer, criterion)
        out = time.time()
        time_avg = (time_avg + (out-start))/2
        avg_loss = sum(losses)/len(losses)
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, correct: {sum(correct)/len(correct):.4f},\tpospair ({sum(correct_pos_pair)}),\tnegpair ({sum(correct_neg_pair)})\tavg-time: {int(time_avg)}s, estimate left: {str(timedelta(seconds=((6005-epoch)*time_avg))).split('.')[0]}")

        LOG.add_scalar("loss/train-avg", avg_loss, epoch)

        with torch.no_grad():
            # Generate a graph of the network
            # When using pytorch geometric, it is a mess, and can be used too see the memory limitations
            # if epoch == 2:
            #     sample_data = next(iter(dataloader))
            #     single_data = sample_data[0].get_example(0).to(device)
            #     LOG.add_graph(model, [(single_data.pos, single_data.edge_index)])
            
            pr_curve_samples = 1023
            toprint = [[],[],[]]  # 3 types of metrics ergo 3 buckets (descriptor rank1, siamese verification, siamese rank1)

            def savefig(fig, dir, name):
                if type(LOG).__name__ != "Dummy":
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    fig.savefig(os.path.join(dir, name), bbox_inches='tight')

            def generate_rank1(metric, dataset_name, experiment_type, tag, print_order, short=True, loss=None):
                loss = f"loss:{loss:.3f}, " if loss else ""
                toprint[print_order].append(f"{experiment_type}-{dataset_name}-{tag}\t{loss}{metric.__str_short__() if short else metric}")
                metric.log_minimal(f"{dataset_name}-{experiment_type}", tag, epoch, LOG)
            
            # TODO revert offload
            def generate_siamese_verification(siam, device, criterion, descriptor_dict, dataset_name, tag, print_order):
                # offload = "frgc" in dataset_name or "bosp" in dataset_name
                offload = "frgc" in dataset_name  # The FRGC dataset is so large that it needs to be offlaoded to the CPU
                loss, metric, preds, labels = metrics.generate_metric_siamese(siam, device, criterion, descriptor_dict, offload)

                full_name = f"{dataset_name}-siamese-verification"
                toprint[print_order].append(f"siamese-verification-{dataset_name}-{tag},\tloss:{loss:.3f}, {metric}")
                metric.log_maximal(full_name, tag, epoch, LOG)
                LOG.add_scalar(f"loss/{full_name}-{tag}", loss, epoch)
                LOG.add_pr_curve(f"{full_name}-pr/{tag}", labels, preds, epoch, num_thresholds=pr_curve_samples)
                #roc_fig =  metrics.generate_roc(labels, preds)

                # Save figure
                #folderpath = os.path.join(logging_dir, logging_name, f"{full_name}-roc-{tag}")
                #savefig(roc_fig, folderpath, f"roc-{epoch}.pdf")
                #LOG.add_figure(f"{full_name}-roc/{tag}", roc_fig, epoch)
                #plt.close(roc_fig)
                
            # Takes some arguments implicit, like model, siamese model, device, criterion
            def generate_log_block(dataset_name, tag, dataloadr, gal_probe_split):
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloadr, device=device)
                gallery_dict, probe_dict = gal_probe_split(descriptor_dict)

                # Descriptor rank1
                metric = metrics.get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict)
                generate_rank1(metric, dataset_name, "descriptor-rank1", tag, print_order=0)

                # Siamese verification
                generate_siamese_verification(siam, device, criterion, descriptor_dict, dataset_name, tag, print_order=1)

                # Siamese Rank1
                loss, metric = metrics.generate_metirc_siamese_rank1(siam, device, criterion, gallery_dict, probe_dict)
                generate_rank1(metric, dataset_name, "siamese-rank1", tag, print_order=2, loss=loss)

                # ROC and CMC
                def generate_cmc_or_roc_fig(combined_fig, type):
                    folderpath = os.path.join(logging_dir, logging_name, f"{dataset_name}-siamese-{type}-combined-{tag}")
                    savefig(combined_fig, folderpath, f"{type}-combined-{epoch}.pdf")
                    LOG.add_figure(f"{dataset_name}-siamese-{type}-combined/{tag}", combined_fig, epoch)
                    plt.close(combined_fig)

                def genrate_cmc_and_roc(cmc_func, roc_func):
                    combined_cmc_fig = cmc_func(descriptor_dict, siam, device)
                    generate_cmc_or_roc_fig(combined_cmc_fig, "cmc")
                    combined_roc_fig, combined_roc_fig_log, verification_rates, auc, fpr_vs_acc_fig, ap = roc_func(descriptor_dict, siam, device)
                    generate_cmc_or_roc_fig(combined_roc_fig, "roc")
                    generate_cmc_or_roc_fig(combined_roc_fig_log, "roc-log")
                    generate_cmc_or_roc_fig(fpr_vs_acc_fig, "fpr-acc-log")
                    LOG.add_scalar(f"{dataset_name}-siamese-01VR/{tag}", verification_rates[0], epoch)
                    LOG.add_scalar(f"{dataset_name}-siamese-1VR/{tag}", verification_rates[1], epoch)
                    LOG.add_scalar(f"{dataset_name}-siamese-auc/{tag}", auc, epoch)
                    LOG.add_scalar(f"{dataset_name}-siamese-ap/{tag}", ap, epoch)

                    _, _, verification_rates, auc, _, ap = evaluation.all_v_all_generate_roc(descriptor_dict, siam, device)
                    LOG.add_scalar(f"{dataset_name}-siamese-all-01VR/{tag}", verification_rates[0], epoch)
                    LOG.add_scalar(f"{dataset_name}-siamese-all-1VR/{tag}", verification_rates[1], epoch)
                    LOG.add_scalar(f"{dataset_name}-siamese-all-auc/{tag}", auc, epoch)
                    LOG.add_scalar(f"{dataset_name}-siamese-all-ap/{tag}", ap, epoch)


                if "bu3dfe" in dataset_name:
                    # if train_on != "bu3dfe": assert len(gallery_dict) == 100
                    genrate_cmc_and_roc(evaluation.bu3dfe_generate_cmc, evaluation.bu3dfe_generate_roc)
                if "bosp" in dataset_name:
                    # if train_on != "bosp": assert len(gallery_dict) == 105
                    genrate_cmc_and_roc(evaluation.bosphorus_generate_cmc, evaluation.bosphorus_generate_roc)
                if "frgc" in dataset_name:
                    # if train_on != "frgc": assert len(gallery_dict) == 466
                    genrate_cmc_and_roc(evaluation.frgc_generate_cmc, evaluation.frgc_generate_roc)
                if "3dface" in dataset_name:
                    genrate_cmc_and_roc(evaluation.face3d_generate_cmc, evaluation.face3d_generate_roc)


            if epoch % default_epoch_per_log == 0:
                model.eval(); siam.eval(); torch.cuda.empty_cache()
                print("Testing on BU3DFE", end="\r")

                if "bu3dfe" in train_on:
                    generate_log_block("bu3dfe", "val",   dataloader_bu3dfe_test,  metrics.split_gallery_set_vs_probe_set_BU3DFE)
                    generate_log_block("bu3dfe", "train", dataloader_bu3dfe_train, metrics.split_gallery_set_vs_probe_set_BU3DFE)
                else:
                    generate_log_block("bu3dfe", "all", dataloader_bu3dfe_all, metrics.split_gallery_set_vs_probe_set_BU3DFE)

            if epoch % default_epoch_per_log == 0:
                model.eval(); siam.eval(); torch.cuda.empty_cache()
                print("Testing on Bosphorus", end="\r")

                if "bosp" in train_on:
                    generate_log_block("bosphorus", "val",   dataloader_bosphorus_test,  metrics.split_gallery_set_vs_probe_set_bosphorus)
                    generate_log_block("bosphorus", "train", dataloader_bosphorus_train, metrics.split_gallery_set_vs_probe_set_bosphorus)
                else:
                    generate_log_block("bosphorus", "all", dataloader_bosphorus_all, metrics.split_gallery_set_vs_probe_set_bosphorus)


            if epoch % default_epoch_per_log == 0:
                model.eval(); siam.eval(); torch.cuda.empty_cache()
                print("Testing on FRGC     ", end="\r")

                if "frgc" in train_on:
                    generate_log_block("frgc", "val",   dataloader_frgc_train,  metrics.split_gallery_set_vs_probe_set_frgc)
                    generate_log_block("frgc", "train", dataloader_frgc_test, metrics.split_gallery_set_vs_probe_set_frgc)
                else:
                    generate_log_block("frgc", "all", dataloader_frgc_all, metrics.split_gallery_set_vs_probe_set_frgc)


            # if epoch % default_epoch_per_log == 0:
            #     model.eval(); siam.eval(); torch.cuda.empty_cache()
            #     print("Testing on 3dface    ", end="\r")
            #     generate_log_block("3dface", "all", dataloader_3dface_all, metrics.split_gallery_set_vs_probe_set_3dface)


            if sum(len(section) for section in toprint) > 0:
                print(" "*20, end="\r")
                for section in toprint:
                    for line in section:
                        print(line)
                torch.cuda.empty_cache()  # Needed to stop memory leak... TODO figure out why metric uses 4gb ish ram, or just allow it
                plt.close("all")  # matplotlib didnt register the plots are correctly closed. Force all plots to close
        
        if epoch % 500 == 0 and type(LOG).__name__ != "Dummy":
            name = f"{logging_dir}/{logging_name}/model-{epoch}.pt"
            name_siam = f"{logging_dir}/{logging_name}/model-siam-{epoch}.pt"
            print(f"Saving {name}")
            torch.save(model.state_dict(), name)
            torch.save(siam.state_dict(), name_siam)

import os
logging_dir = "logging-siamese-1905-namechange-trash"
logging_name = ""
def ask_for_writer(dataloader, optimizer):
    global logging_name
    batch_size = dataloader.batch_size
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    import inspect
    function_name = inspect.stack()[1][3]

    from datetime import date
    dato = date.today().strftime("%Y-%m-%d")

    import sys
    if len(sys.argv) > 1:
        extra = sys.argv[1]
    else:
        extra = input("Siste endringer: ")

    if extra == "q" or not extra:
        print("Logging disabled")
        import random
        logging_name = "q" + str(random.randint(0, 1000))
        class Dummy:
            def add_scalar(*args, **kwargs):  # Dummy function that can take any argument
                return
            def __getattr__(self, attr):  # Overwrite python class get func
                return self.add_scalar
        return Dummy()

    navn = f"{dato}_lr{lr:1.0e}_batchsize{batch_size}_{extra.replace(' ', '-')}"
    logging_name = navn
    import os
    from torch.utils.tensorboard import SummaryWriter
    WRITER = SummaryWriter(log_dir=os.path.join(logging_dir, navn))
    print("Logging enabled: " + navn)
    print("Full path: " + logging_dir + "/" + navn)
    return WRITER

if __name__ == '__main__':
    print(f"Cuda: {torch.cuda.is_available()}")
    # print("test1"); test_1_regular_poitnet()
    # print("test2"); test_2_pointnet_triplet_loss()
    # print("test3"); test_3_poitnet_softmax()
    # print("test4"); test_4_convnet_softmax()
    # print("test5"); test_5_convnet_triplet()
    # print("test6"); test_6_softmax_embeddings()
    # print("test7"); test_7_softmax_embeddings2()
    print("test8"); test_8_convnet_triplet()
