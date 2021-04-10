import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

torch.manual_seed(1)
torch.cuda.manual_seed(1)

from tqdm import tqdm

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

        self.sa1_module = SAModule(0.5, 0.2/3, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4/3, MLP([128 + 3, 128, 128, 256]))
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
        x = F.normalize(x, dim=-1, p=2)  # L2 Normalization tips
        return x
        # return F.log_softmax(x, dim=-1)  # remember correct out shape

# Test 1

def train(epoch, dataloader, optimizer):
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
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def test_1_regular_poitnet():
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
        losses = train(epoch, dataloader, optimizer)
        avg_loss = sum(losses)/len(losses)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Test: {:.4f}, Avg-loss {:.4f}'.format(epoch, test_acc, avg_loss))


# Test 2 - poitnet++ with triplet loss
import torch_geometric.data.batch as geometric_batch
import onlineTripletLoss
import metrics
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

        self.conv3 = GCNConv(in_channels=94, out_channels=128)
        self.batch3 = BatchNorm(in_channels=self.conv3.out_channels)

        # self.conv4 = GCNConv(in_channels=256, out_channels=512)
        # self.batch4 = BatchNorm(in_channels=self.conv4.out_channels)

        # self.conv5 = GCNConv(in_channels=256, out_channels=512)
        # self.batch5 = BatchNorm(in_channels=self.conv5.out_channels)

        # self.pooling1 = TopKPooling(in_channels=1, ratio=1024)

        self.fc0 = Linear(128, 128)

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
        # print("d", data)
        # print("d0", data.get_example(0))
        # print("pre scatter", x.shape)
        x = scatter(x, batch, dim=0)  # Unsure if the correct
        # print("post scatter", x.shape)
        # return scatter(x, batch, dim=0, dim_size=x.shape[0], reduce='add') ?
        # x = global_max_pool(x, batch)

        x = self.activation(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        # x = F.normalize(x, dim=-1, p=2)  # L2 Normalization tips
        return x
        # return F.log_softmax(x, dim=-1)


# Test 5 - convnet with triplet loss
import torch_geometric.data.batch as geometric_batch
import metrics
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
            dist_a_p.append(torch.dist(descritors[0][0].to("cpu"), descritors[0][1].to("cpu"), p=2).item())
            dist_a_n.append(torch.dist(descritors[1][0].to("cpu"), descritors[0][0].to("cpu"), p=2).item())
            
            lengths.append(torch.norm(descritors[0][0], 2))
            max_losses.append(max_loss.item())
            max_dist_a_ps.append(max_dist_a_p.item())
            min_dist_a_ns.append(min_dist_a_n.item())
    # return losses, dist_a_p, dist_a_n
    return losses, dist_a_p, dist_a_n, lengths, max_losses, max_dist_a_ps, min_dist_a_ns

import datasetBU3DFE
import math
def test_5_convnet_triplet():
    import reduction_transform
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True),T.NormalizeScale()])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    
    start_epoch = 1  # re-written if starting from a loaded save

    # DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    # BU3DFE_HELPER = datasetBU3DFE.BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    # dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    import pickle

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
    dataloader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    
    dataloader_val = DataLoader(dataset=train_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    dataloader_test = DataLoader(dataset=test_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestNet55_desc().to(device)

    # print("Loading save")
    # model.load_state_dict(torch.load("./Testnet55_desc-triplet-128desc-500.pt"))
    # start_epoch += 500

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-25)
    # loss = ||ap|| - ||an|| + margin.  neg loss => ||an|| >>> ||ap||, at least margin over

    margin = 0.2

    # critering used in naive approach
    criterion = torch.nn.TripletMarginLoss(margin=margin)  # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss   mean or sum reduction possible

    LOG = ask_for_writer(dataloader, optimizer)
    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    for epoch in range(start_epoch, 500):
        losses, dist_a_p, dist_a_n, lengths, max_losses, max_dist_a_ps, min_dist_a_ns = train5(epoch, model, device, dataloader, optimizer, margin, criterion)
        # losses, dist_a_p, dist_a_n = train5(epoch, model, device, dataloader, optimizer, margin, criterion)
        avg_loss = sum(losses)/len(losses)
        dist_a_p = sum(dist_a_p)/len(dist_a_p)
        dist_a_n = sum(dist_a_n)/len(dist_a_n)
        # lengths = sum(lengths)/len(lengths)
        #dist_a_n = avg_loss - 0.2 - dist_a_p    # loss = margin + a_p + a_n => a_n = loss - margin - ap
        print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}, avg_desc_length: {(sum(lengths)/len(lengths)):.2f}, max_loss: {max(max_losses):.4f}, max_dist_a_p: {max(max_dist_a_ps):.4f}, min_dist_a_n: {min(min_dist_a_ns):.4f}")
        # print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, dist_a_p: {dist_a_p:.4f}, dist_a_n: {dist_a_n:.4f}")
        z = 0.00001

        LOG.add_scalar("Loss/train-avg", avg_loss, epoch)
        LOG.add_scalar("Distance/anchor-positive", dist_a_p, epoch)
        LOG.add_scalar("Distance/anchor-negative", dist_a_n, epoch)
        try:
            LOG.add_scalar("Loss/train-max", max(max_losses), epoch)
            LOG.add_scalar("Distance/max-anchor-positive", max(max_dist_a_ps), epoch)
            LOG.add_scalar("Distance/min-anchor-negative", min(min_dist_a_ns), epoch)
            LOG.add_scalar("Distance/length", min(min_dist_a_ns), epoch)
        except:
            pass

        if dist_a_p < z and dist_a_n < z and margin - 10*z < avg_loss < margin + 10*z:
            print("Stopping due to collapse of descriptors"); import sys; sys.exit(-1)
        if math.isnan(avg_loss):
            print("Stopping due to nan"); import sys; sys.exit(-1)

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                # descriptor_dict = metrics.data_dict_to_descriptor_dict(model=model, device=device, data_dict=cfg.DATASET_HELPER.get_cached_dataset(), desc="Evaluation/Test", leave_tqdm=False)
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_test, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
                print("RANK-1-valdata", metric.__str_short__())
                for m in ["tp", "fp", "accuracy"]:
                    LOG.add_scalar("metric-" + m + "/val", getattr(metric, m), epoch)
                
                descriptor_dict = metrics.generate_descriptor_dict_from_dataloader(model=model, dataloader=dataloader_val, device=device)
                metric = metrics.get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
                print("RANK-1-traindata", metric.__str_short__())
                for m in ["tp", "fp", "accuracy"]:
                    LOG.add_scalar("metric-" + m + "/train", getattr(metric, m), epoch)

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
        self.lin4 = Lin(128, 100)  # Softmax, 100 possible different classes, but not all are used, is that bad?

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


# ask_for_writer(dataloader, optimizer)
def ask_for_writer(dataloader, optimizer):
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
        class Dummy:
            def add_scalar(*args):
                return
        return Dummy()

    navn = f"{str(dato)}_lr{lr:1.0e}_batchsize{batch_size}_{extra.replace(' ', '-')}"
    import os
    from torch.utils.tensorboard import SummaryWriter
    WRITER = SummaryWriter(log_dir=os.path.join("logging", navn), max_queue=20)
    print("Logging enabled: " + navn)
    return WRITER

if __name__ == '__main__':
    print(f"Cuda: {torch.cuda.is_available()}")
    # print("test1"); test_1_regular_poitnet()
    # print("test2"); test_2_pointnet_triplet_loss()
    # print("test3"); test_3_poitnet_softmax()
    # print("test4"); test_4_convnet_softmax()
    print("test5"); test_5_convnet_triplet()
    # print("test6"); test_6_softmax_embeddings()