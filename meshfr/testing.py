import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, Linear as Lin, ReLU, BatchNorm1d as BN


from torch_geometric.nn import PointConv, fps, radius, global_max_pool  # Get the needed NN for pointnet++
from torch_geometric.nn import GCNConv, BatchNorm, global_max_pool  # Get the needed NN for convnet
from torch_geometric.data.batch import Batch  # Get the type 
from torch_geometric.data.data import Data    # Get the type 
import torch_geometric.transforms as T
import torch_geometric.data.batch as geometric_batch

#from torch_geometric.data import DataLoader  # Instead of this, use modified dataloader to not throw away data 
from meshfr.datasets.datasetGeneric import DataLoader


import meshfr.evaluation.metrics as metrics
import meshfr.evaluation.realEvaluation as evaluation
from meshfr.datasets.datasetGeneric import GenericDataset, ExtraTransform
import meshfr.datasets.datasetBU3DFEv2 as datasetBU3DFEv2
import meshfr.datasets.datasetBosphorus as datasetBosphorus
from meshfr.datasets.datasetFRGC import get_frgc_dict
import meshfr.datasets.reduction_transform as reduction_transform  # Homemade transformations
import meshfr.tripletloss.onlineTripletLoss as onlineTripletLoss

import matplotlib.pyplot as plt  # Used to fix memory leak
import numpy as np
import random 
import time 
from datetime import timedelta


torch.manual_seed(1)
torch.cuda.manual_seed(1)


# Pointnet
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


class PointnetPP(torch.nn.Module):
    def __init__(self):
        super(PointnetPP, self).__init__()
        torch.manual_seed(1)

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
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




class TestNet55_descv2(torch.nn.Module):
    def __init__(self):
        super(TestNet55_descv2, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.embeddings = True  # Default return embeddings
        self.fcSoftmax = Linear(128, 105)

        self.activation = ReLU()

        # org: 3,64,128,128,256,512,pool,512,256,256,128
        self.conv1 = GCNConv(in_channels=3, out_channels=16)
        self.batch1 = BatchNorm(in_channels=self.conv1.out_channels)

        self.conv11 = GCNConv(in_channels=16, out_channels=32)
        self.batch11 = BatchNorm(in_channels=self.conv11.out_channels)

        # 
        self.conv12 = GCNConv(in_channels=32, out_channels=64)
        self.batch12 = BatchNorm(in_channels=self.conv12.out_channels)

        # 
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
        x = F.normalize(x, dim=-1, p=2)  # L2 Normalization tips

        if self.embeddings:
            return x
        else:
            x = self.fcSoftmax(self.activation(x))
            return F.log_softmax(x, dim=-1)

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


# Distance based siamese
class Siamese_part_distance(torch.nn.Module):
    def __init__(self):
        super(Siamese_part_distance, self).__init__()

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        distances =  -torch.norm(x1 - x2, p=2, dim=-1)
        return torch.squeeze(distances)


def train8_triplet(model, device, dataloader, optimizer):
    model.train()

    losses = []
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
            # id = batch_all.id[i].item()
            id = batch_all.dataset_id[i]  # Use string name instead of idx id
            if id in dic_descriptors:
                dic_descriptors[id].append(descritors[i])
            else:
                dic_descriptors[id] = [descritors[i]]
        descritors = list(dic_descriptors.values())
        
        # Alegedly would fix "RuntimeError: received 0 items of ancdata"
        del batch
        del batch_all

        all = []
        labels = []  # indencies for all, eks [0,0,0,1,1,2,2,3,3]
        for ident, listt in enumerate(descritors):
            all += listt
            labels += [ident] * len(listt)

        all = torch.stack(all).to(device)
        labels = torch.tensor(labels).to(device)

        loss, max_loss, max_dist_a_p, min_dist_a_n = onlineTripletLoss.batch_hard_triplet_loss(labels=labels, embeddings=all, margin=0.2, device=device)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses.append(loss.item())
            lengths.append(torch.norm(descritors[0][0], 2))
            try: 
                max_losses.append(max_loss.item())
                max_dist_a_ps.append(max_dist_a_p.item())
                min_dist_a_ns.append(min_dist_a_n.item())
            except:
                pass
        return losses, lengths, max_losses, max_dist_a_ps, min_dist_a_ns

def train8_softmax(model, device, dataloader, optimizer):
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

        # Uses "illegal" id. DO NOT MIX DATASETS
        loss = F.nll_loss(output, batch_all.id)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses.append(loss.item())
            total_samples = output.shape[0]
            correct_samples = output.max(1)[1].eq(batch_all.id).sum().item()
        return losses, total_samples, correct_samples


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
        # This may be un-needed, but works
        dic_descriptors = {}
        for i in range(len(batch_all.id)):
            # id = batch_all.id[i].item()
            id = batch_all.dataset_id[i]  # Use string name instead of idx id
            if id in dic_descriptors:
                dic_descriptors[id].append(descritors[i])
            else:
                dic_descriptors[id] = [descritors[i]]
        descritors = list(dic_descriptors.values())
        
        # Alegedly would fix "RuntimeError: received 0 items of ancdata"
        del batch
        del batch_all

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


def test_8_convnet_triplet():
    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    # POST_TRANSFORM_Extra = T.Compose([])
    POST_TRANSFORM_Extra = T.Compose([
        # reduction_transform.RandomTranslateScaled(0.0075),
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

    run_on_device = "cuda"
    assert run_on_device in ["cuda", "cpu"]
    train_on = "bosp"
    assert train_on in ["bu3dfe", "bosp", "frgc", "bosp+frgc", "bu3dfe+bosp+frgc", "bu3dfe+bosp"]
    siamese_network_type = "triplet"
    assert siamese_network_type in ["siamese", "triplet", "softmax"]
    assert siamese_network_type != "softmax" or "+" in train_on  # DO NOT MIX SOFTMAX AND DATASETS
    # When using softmax, make sure that the fcSoftmax is correctly configured

    # Global properties
    pickled = True
    force = False
    sample = "bruteforce"  # 2pass, bruteforce, all or random
    sample_size = [1024*2, 1024*8][0]
    num_workers_train = 10  # Maximum amount of dataloaders, may be overwritten if the batch size is less
    # torch.multiprocessing.set_sharing_strategy('file_system')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # for the dataloader. As it is in memory, a high number is not needed, set to 0 if file desc errors https://pytorch.org/docs/stable/data.html
    # Alt,  check out   lsof | awk '{ print $2; }' | uniq -c | sort -rn | head
    # and               ulimit -n 4096
    def dload(dataset, batch_size, predicable, num_workers=0):
        num_workers = min(num_workers, batch_size)
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

    frgc_path = "/lhome/haakowar/Downloads/FRGCv2/Data/"
    frgc_fall_2003_dict = get_frgc_dict(frgc_path + "Fall2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-fall2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
    frgc_spring_2003_dict = get_frgc_dict(frgc_path + "Spring2003range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2003_cache-2048-new.p", sample=sample, sample_size=sample_size)
    frgc_spring_2004_dict = get_frgc_dict(frgc_path + "Spring2004range", pickled=pickled, force=force, picke_name="/tmp/FRGCv2-spring2004_cache-2048-new.p", sample=sample, sample_size=sample_size)
    dataset_frgc_fall_2003 = GenericDataset(frgc_fall_2003_dict, POST_TRANSFORM)
    dataset_frgc_spring_2003 = GenericDataset(frgc_spring_2003_dict, POST_TRANSFORM)
    dataset_frgc_spring_2004 = GenericDataset(frgc_spring_2004_dict, POST_TRANSFORM)
    dataset_frgc_train = torch.utils.data.ConcatDataset([dataset_frgc_spring_2003])  # As per doc, Spring2003 is train, rest is val
    dataset_frgc_train = ExtraTransform(dataset_frgc_train, POST_TRANSFORM_Extra)
    dataset_frgc_test = torch.utils.data.ConcatDataset([dataset_frgc_fall_2003, dataset_frgc_spring_2004])  # As per doc, Spring2003 is train, rest is val
    dataloader_frgc_train = dload(dataset_frgc_train, batch_size=2, predicable=True)
    dataloader_frgc_test = dload(dataset_frgc_test, batch_size=2, predicable=True)
    dataset_frgc_all = torch.utils.data.ConcatDataset([dataset_frgc_fall_2003, dataset_frgc_spring_2003, dataset_frgc_spring_2004])
    dataloader_frgc_all = dload(dataset_frgc_all, batch_size=5, predicable=True)
    if train_on == "frgc":
        dataloader = dload(dataset_frgc_train, batch_size=50, predicable=False, num_workers=num_workers_train)

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
    
    if train_on == "bu3dfe+bosp+frgc":
        dataset_bu3dfe_frgc_bosp_train = torch.utils.data.ConcatDataset([bu3dfe_train_set, bosphorus_train_set, dataset_frgc_train])
        dataloader = dload(dataset_bu3dfe_frgc_bosp_train, batch_size=5, predicable=False, num_workers=num_workers_train)

    if train_on == "bu3dfe+bosp":
        dataset_frgc_bosp_train = torch.utils.data.ConcatDataset([bu3dfe_train_set, bosphorus_train_set])
        dataloader = dload(dataset_frgc_bosp_train, batch_size=10, predicable=False, num_workers=num_workers_train)

    if run_on_device == "cuda":
        assert torch.cuda.is_available()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(os.cpu_count())
        print(f"os: {os.cpu_count()}, torch {torch.get_num_threads()}")
        

    # Load the model
    print(f"PSA: using {device}")
    model = TestNet55_descv2().to(device)
    # model = PointnetPP().to(device)  # To test with pointnet as a feature extractor
    if siamese_network_type == "siamese":
        siam = Siamese_part().to(device)
        criterion = torch.nn.BCELoss()
    elif siamese_network_type == "triplet" or siamese_network_type == "softmax":
        siam = Siamese_part_distance().to(device)
        criterion = None

    # print("Loading save")
    # model.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-17_lr1e-03_batchsize6_testing-bosp-norm-t001-r5-axis012/model-500.pt"))
    # siam.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-17_lr1e-03_batchsize6_testing-bosp-norm-t001-r5-axis012/model-siam-500.pt"))
    # start_epoch += 1000

    # print("Loading save")
    # model.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-13_lr1e-03_batchsize10_testing-bu3dfe-norm-translate001-rotate5-axis012/model-6000.pt", map_location=device))
    # siam.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-13_lr1e-03_batchsize10_testing-bu3dfe-norm-translate001-rotate5-axis012/model-siam-3000.pt", map_location=device))
    # start_epoch += 6000
    
    # print("Loading save")
    # model.load_state_dict(torch.load("./logging-siamese-1905-namechange-trash/2021-06-23_lr1e-03_batchsize20_bu3dfe-norm-translate001-rotate5-axis012-TRIPLET-fresh-small-net-smaller2/model-500.pt", map_location=device))
    # start_epoch += 500
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(siam.parameters()), lr=1e-3)

    LOG = ask_for_writer(dataloader, optimizer)
    print(f"dataloader_batch: {dataloader.batch_size}, optimizer: {optimizer.state_dict()['param_groups'][0]['lr']}")
    time_avg = 1
    max_epochs = 6005
    for epoch in range(start_epoch, max_epochs):
        model.train()
        siam.train()
        start = time.time()
        if siamese_network_type == "siamese":
            losses, correct, correct_pos_pair, correct_neg_pair = train8_sia(model, siam, device, dataloader, optimizer, criterion)
        elif siamese_network_type == "triplet":
            losses, lengths, max_losses, max_dist_a_ps, min_dist_a_ns = train8_triplet(model, device, dataloader, optimizer)
        elif siamese_network_type == "softmax":
            model.embeddings = False
            losses, total_samples, correct_samples = train8_softmax(model, device, dataloader, optimizer)
            model.embeddings = True
        out = time.time()
        time_avg = (time_avg + (out-start))/2
        avg_loss = sum(losses)/len(losses)
        if siamese_network_type == "siamese":
            print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, correct: {sum(correct)/len(correct):.4f},\tpospair ({sum(correct_pos_pair)}),\tnegpair ({sum(correct_neg_pair)})\tavg-time: {int(time_avg)}s, estimate left: {str(timedelta(seconds=((max_epochs-epoch)*time_avg))).split('.')[0]}")
        elif siamese_network_type == "triplet":
            print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, avg_desc_length: {(sum(lengths)/len(lengths)):.2f}, max_loss: {max(max_losses):.4f}, max_dist_a_p: {max(max_dist_a_ps):.4f}, min_dist_a_n: {min(min_dist_a_ns):.4f}, \tavg-time: {int(time_avg)}s, estimate left: {str(timedelta(seconds=((max_epochs-epoch)*time_avg))).split('.')[0]}")
        elif siamese_network_type == "softmax":
            print(f"Epoch:{epoch}, avg_loss: {avg_loss:.4f}, total_samples: {total_samples}, correct_samples: {correct_samples}\tavg-time: {int(time_avg)}s, estimate left: {str(timedelta(seconds=((max_epochs-epoch)*time_avg))).split('.')[0]}")
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
                # generate_siamese_verification(siam, device, criterion, descriptor_dict, dataset_name, tag, print_order=1)

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
                    # These are neutral vs all. (or first vs all)
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
