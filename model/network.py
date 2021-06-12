import torch
from torch_geometric.nn import TopKPooling, GCNConv, BatchNorm
from torch import nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)

from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
def max_pool_neighbor_x(x, edge_index, flow='source_to_target'):
    r"""Max pools neighboring node features, where each feature in
    :obj:`data.x` is replaced by the feature value with the maximum value from
    the central node and its neighbors.
    """

    edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

    row, col = edge_index
    row, col = (row, col) if flow == 'source_to_target' else (col, row)

    x = scatter(x[row], col, dim=0, dim_size=x.shape[0], reduce='max')
    # if batch is None:
    #             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    return x


def global_max_pool(x, batch, size = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=x.shape[0], reduce='max')

def max_pool_2(x, edge_index=None):
    r, _ = torch.max(x, dim=1)
    # print("r", r.shape)
    return r.unsqueeze(1)
    # row, col = edge_index
    # print("pool", x.shape)
    # print("col", col.shape)
    # print("row", row.shape)
    # return scatter(x[row], col, dim=0, dim_size=1, reduce='max')

class MaxPool23(nn.Module):
    def __init__(self):
        super(MaxPool23, self).__init__()

    def forward(self, x):
        # print("x", x.shape)  torch.Size([7805, 2048])
        r, _ = torch.max(x, dim=0)
        # print("r", r.shape)   torch.Size([2048])
        return r.unsqueeze(1)

class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        # Network note:
        #   Conv1 + activation
        #   Pool1
        #   Conv2 + activation
        #   Conv3 + activation
        #   Pool2 + activation
        #   Flatten
        #   FC

        # 63- FC+acti
        # 64- Pool1 512 -> 4096
        # 65- FC - acti, mer epochs
        #   Epoch 79 tp=61, fp=0, tn=19800, fn=39, acc=0.9980, recall=0.6100, f1=0.7578, FRR=0.3900, FAR=0.0000
        #   Greit
        # 66- Til pooling, til conv, -acti etter conv2, større FC
        #   Prenote: endret for mye? -- vanskelig å vite hva som endrer
        #   doingnote: Ser ikke ut som å lære. For høy LR?
        #   Stoppet tidlig
        #   Epoch 37 tp=3, fp=0, tn=19800, fn=97, acc=0.9951, recall=0.0300, f1=0.0583, FRR=0.9700, FAR=0.0000
        # 67- 66 bare lr på 5*10^-5
        #   Epoch 19 tp=2, fp=0, tn=19800, fn=98, acc=0.9951, recall=0.0200, f1=0.0392, FRR=0.9800, FAR=0.0000
        #   Crasjet pga div by 0 (precition4)
        #   Gjør shit, slettes
        # 68- Reduser som faen, bort med conv 4, 5, og pool 2
        #  Går da fra maks -> 16*4k -> 32*512
        # 68 - back til vanlig: 3*M -> 16*M -> 16*4k -> 32*4k -> 64*4k -> 64*128 -> flatten -> 128
        # 72- default, expanded + reduced
        # 73- la til conv2 etter conv1, dårligere resultat

        # Ny:
        # 126 Pooling 2:  256 nodes, 32 channels (default). 256 output nodes  (Økt siste pooling og output)
        # 127: conv3 out fra 64 til 128. Propagaste det
        # 128 mange flere params conv 2 3,  høyere lr
        # 129 øke channels conv 1 lignende pointnetcnn

        self.activation = nn.LeakyReLU()

        self.conv1 = GCNConv(in_channels=3, out_channels=64)
        self.pooling1 = TopKPooling(in_channels=64, ratio=4096)

        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.conv3 = GCNConv(in_channels=128, out_channels=256)
        self.pooling2 = TopKPooling(in_channels=256, ratio=256)

        self.flatten = nn.Flatten(start_dim=0)  # Special start dim as it is not yet batched
        self.fc1 = nn.Linear(256*256, 512)


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

        # Flatten type
        x = x.transpose(0, 1)  # transpose to regular structure of [dims, nodes]
        x = self.flatten(x)
        x = self.fc1(x)

        return x

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



class TestNet2(torch.nn.Module):
    def __init__(self):
        super(TestNet2, self).__init__()

        # Input data
        # 6 conv, Batchnorm?, relu
        # 3 -> 64, 128, 2048, maxpool, 256

        self.activation = nn.LeakyReLU()

        self.pooling1 = TopKPooling(in_channels=1, ratio=2048)

        self.conv1 = GCNConv(in_channels=3, out_channels=64)
        self.batch1 = BatchNorm(in_channels=64)

        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.batch2 = BatchNorm(in_channels=128)

        self.conv3 = GCNConv(in_channels=128, out_channels=256)
        self.batch3 = BatchNorm(in_channels=256)

        self.conv4 = GCNConv(in_channels=256, out_channels=512)
        self.batch4 = BatchNorm(in_channels=512)

        self.conv5 = GCNConv(in_channels=512, out_channels=1024)
        self.batch5 = BatchNorm(in_channels=1024)

        # --- max pool

        self.conv6 = GCNConv(in_channels=1, out_channels=8)
        self.batch6 = BatchNorm(in_channels=8)

        self.conv7 = GCNConv(in_channels=8, out_channels=64)
        self.batch7 = BatchNorm(in_channels=64)

        self.conv8 = GCNConv(in_channels=64, out_channels=128)
        self.batch8 = BatchNorm(in_channels=128)

        self.conv8 = GCNConv(in_channels=64, out_channels=2048)
        self.batch8 = BatchNorm(in_channels=2048)

        self.maxpool = max_pool_2  # max pool the node channel to 1
        self.maxpool23 = MaxPool23()

        self.flatten = nn.Flatten(start_dim=0)  # Special start dim as it is not yet batched
        self.fc1 = nn.Linear(1024, 1024)  # One value per node, got better resuts when going from 512 to 1024

        self.fcprepool1 = nn.Linear(1024, 256)
        self.fcprepool2 = nn.Linear(2048, 256)

        self.activation2 = nn.Tanh()


    def forward(self, data):
        pos, edge_index = data.pos, data.edge_index
        x = pos

        x = self.conv1(x, edge_index)
        # x = self.batch1(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        # x = self.batch2(x)
        x = self.activation(x)

        x = self.conv3(x, edge_index)
        # x = self.batch3(x)
        x = self.activation(x)

        x = self.conv4(x, edge_index)
        # x = self.batch4(x)
        x = self.activation(x)

        x = self.conv5(x, edge_index)
        # x = self.batch5(x)
        x = self.activation(x)

        # # print(x.shape)  # [Node, channels]
        # x = self.fcprepool1(x)  # possibly faster convergance / better results from this
        # x = self.activation(x)
        #
        # x = self.maxpool(x, edge_index)
        #
        # x = self.conv6(x, edge_index)
        # # x = self.batch6(x)
        # x = self.activation(x)
        #
        # x = self.conv7(x, edge_index)
        # # x = self.batch7(x)
        # x = self.activation(x)
        #
        # x = self.conv8(x, edge_index)
        # # x = self.batch8(x)
        # x = self.activation(x)

        # x = self.fcprepool2(x)  # possibly slower convergance / slower results from this
        # x = self.activation(x)

        #x = self.maxpool(x, edge_index)
        x = self.maxpool23(x)

        #x, edge_index, edge_attr, batch, perm, score = self.pooling1(x, edge_index)  # Best result when having it last


        # print(1, x.shape)
        # Flatten type
        # Wait, why
        # x = x.transpose(0, 1)  # transpose to regular structure ([nodes, dims] -> [dims, nodes])
        # print(2, x.shape)
        x = self.flatten(x)
        # print(3, x.shape)
        x = self.fc1(x)
        # x = self.activation2(x)  # possibly slower convergance from this

        return x

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


class TestNetTryMode(torch.nn.Module):
    def __init__(self, conv, pool):
        super(TestNetTryMode, self).__init__()

        self.conv_default = conv
        self.pool_default = pool

        self.activation = nn.LeakyReLU()

        self.conv1 = self.conv_default(in_channels=3, out_channels=16)
        self.pooling1 = self.pool_default(in_channels=16, ratio=4096)

        self.conv2 = self.conv_default(in_channels=16, out_channels=32)
        self.conv3 = self.conv_default(in_channels=32, out_channels=32)
        self.pooling2 = self.pool_default(in_channels=32, ratio=128)

        self.flatten = nn.Flatten(start_dim=0)  # Special start dim as it is not yet batched
        self.fc1 = nn.Linear(32*128, 128)


    def forward(self, data):
        pos, edge_index = data.pos, data.edge_index

        x = self.conv1(pos, edge_index)
        x = self.activation(x)
        output = self.pooling1(x, edge_index)
        x, edge_index = output[0], output[1]

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)

        output = self.pooling2(x, edge_index)
        x, edge_index = output[0], output[1]

        # Flatten type
        x = x.transpose(0, 1)  # transpose to regular structure of [dims, nodes]
        x = self.flatten(x)
        x = self.fc1(x)

        return x

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



# The model described in the reliminary report
# DO NOT USE, as it has a lot of limitations
# Mostly as a test to see if the current re-written code gets the same result
class PrelimNet(torch.nn.Module):
    def __init__(self):
        super(PrelimNet, self).__init__()

        # from gcn_conv_test import GCNConv

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
        # print(x.shape) [5850, 10]
        # x = x.transpose(0, 1)  # transpose to regular structure of [dims, nodes]
        # print(x.shape) [10, 5850]
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x

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




class TestNet3(torch.nn.Module):
    def __init__(self):
        super(TestNet3, self).__init__()
        print("Loading model3")

        # Input data
        # 6 conv, Batchnorm?, relu
        # 3 -> 64, 128, 2048, maxpool, 256

        activation = nn.LeakyReLU()
        pool = TopKPooling
        bnorm = BatchNorm
        conv = GCNConv

        def conv_bn_relu(inn, out):
            output = []
            output.append(("conv", conv(in_channels=inn, out_channels=out)))
            output.append(("x", bnorm(in_channels=out)))
            output.append(("x", activation))
            return output

        self.layers = []

        # Input Transformation Network
        #layers += conv_bn_relu(3, 64)
        #layers += conv_bn_relu(64, 128)
        #layers += conv_bn_relu(128, 2048)
        ## TODO maxpool?
        #layers.append(("x", nn.Linear(2048, 256)))

        # Forward network 1
        self.layers += conv_bn_relu(3, 64)
        self.layers += conv_bn_relu(64, 64)

        # Feature Transformation network

        # Forward network 2
        self.layers += conv_bn_relu(64, 64)
        self.layers += conv_bn_relu(64, 128)
        self.layers += conv_bn_relu(128, 2048)

        # Maxpool
        self.layers.append(("x", MaxPool23()))  # Switch maxpool? Instead of going Nx2048 -> Nx1, do Nx2048 -> 1x2048?


        # Dense BN Relu
        # NOT AS IN THE PAPER
        self.layers.append(("x", nn.Flatten(start_dim=0)))   # Special start dim as it is not yet batched
        self.layers.append(("x", nn.Linear(2048, 512)))
        self.layers.append(("x", nn.BatchNorm1d(512)))
        self.layers.append(("x", activation))

        # Output
        self.layers.append(("x", nn.Linear(512, 4096)))
        self.layers.append(("x", nn.BatchNorm1d(4096)))

        self.inputs = [x[0] for x in self.layers]
        self.layers = nn.ModuleList([x[1] for x in self.layers])

    def forward(self, data):
        if isinstance(data, Batch):
            batch = data.batch
        pos, edge_index = data.pos, data.edge_index
        x = pos

        for input, layer in zip(self.inputs, self.layers):
            if input == "x":
                x = layer(x)
            if input == "conv":
                x = layer(x, edge_index)
            if input == "pool-batch":
                x = layer(x, edge_index, batch=batch)
        return x

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



from torch_geometric.nn import PointConv

class TestPoint(torch.nn.Module):
    def __init__(self):
        super(TestPoint, self).__init__()

        self.activation = nn.LeakyReLU()

        self.conv1 = PointConv()


    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x = self.conv1(x=x, pos=pos, edge_index=edge_index)
        x = self.activation(x)

        return x

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

if __name__ == "__main__":
    model = TestNet()
    print(model.short_rep())