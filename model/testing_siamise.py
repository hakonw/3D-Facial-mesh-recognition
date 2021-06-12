# import os.path as osp

import torch
# import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, Linear as Lin, ReLU, BatchNorm1d as BN
# from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
# from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from torch_scatter import scatter
from torch_geometric.nn import GCNConv, BatchNorm, TopKPooling
# import torch_geometric.data.batch as geometric_batch
import onlineTripletLoss
import metrics
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

class Siamise(torch.nn.Module):
    def __init__(self):
        super(Siamise, self).__init__()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.activation = ReLU()

        self.fc0 = Linear(256, 128)

        self.fc1 = Linear(128, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 100)


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
            if True:
                max_losses.append(max_loss.item())
                max_dist_a_ps.append(max_dist_a_p.item())
                min_dist_a_ns.append(min_dist_a_n.item())
    if len(max_losses) == 0:
        return losses, dist_a_p, dist_a_n
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
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
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
        #     name = f"./Testnet55_desc-siamise-{epoch}.pt"
        #     print(f"Saving {name}")
        #     torch.save(model.state_dict(), name)

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
    # print("test1"); test_1_base_convnet_siamise()
