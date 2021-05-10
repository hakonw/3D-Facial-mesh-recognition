from torch_geometric.data import Data
import torch_geometric.utils
import torch
from tqdm import tqdm
import onlineTripletLoss

#
# Processing of data before metrics
#

import setup

@torch.no_grad()  # This function disables autograd, so no training can be done on the data
def single_data_to_descriptor(model, device, data):
    data = setup.Datasets.POST_TRANSFORM(data)
    data.to(device)
    # Assuming single data, ergo dont need to do .to_data_list()
    descriptor = model(data)
    descriptor.to("cpu")  # Results are returned to the cpu
    return descriptor


# Assumes dict: ID -> name -> Data
# : dict[str, dict[str, [Data]]
def data_dict_to_descriptor_dict(model, device, data_dict, desc="Evaluation", leave_tqdm=True):
    descriptor_dict = {}
    model.eval()

    for key, data_list in tqdm(data_dict.items(), desc=desc, leave=leave_tqdm):
        assert isinstance(data_list, dict)
        desc_dict = {}
        for name, data in data_list.items():  # Keep the file metatadata if needed later
            desc_dict[name] = single_data_to_descriptor(model, device, data.clone())

        descriptor_dict[key] = desc_dict
    model.train()
    return descriptor_dict


#
# Helper functions
#

def distance(descriptor1, descriptor2, p_norm=2):
    return torch.dist(descriptor1, descriptor2, p=p_norm)

from dataclasses import dataclass
import dataclasses
@dataclass
class BaseMetric:
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_instance(cls, instance):
        return cls(**dataclasses.asdict(instance))


@dataclass
class ScoreMetric(BaseMetric):
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    FRR: float = None  # (false neg rate) FRR false reject
    FAR: float = None  # (false pos rate) FAR accept rate

    def __str__(self):
        return f"ScoreMetric(tp={self.tp}, fp={self.fp}, tn={self.tn}, fn={self.fn}, " + \
               f"acc={self.accuracy:.4f}, preci={self.precision:.3f} recall={self.recall:.3f}, f1={self.f1:.3f}, " + \
               f"FRR={self.FRR:.3f}, FAR={self.FAR:.3f})"

    def __str_short__(self):
        return f"ScoreMetric(tp={self.tp}, fp={self.fp}, acc={self.accuracy:.4f})"

    def log_minimal(self, name, tag, epoch, logger):
        for m in ["tp", "fp", "accuracy"]:
            logger.add_scalar(f"metric-{name}-{m}/{tag}", getattr(self, m), epoch)

    def log_maximal(self, name, tag, epoch, logger):
        metric_dict = dataclasses.asdict(self)
        for metric_key, metric_value in metric_dict.items():
            logger.add_scalar(f"metric-{name}-{metric_key}/{tag}", metric_value, epoch)

# TODO make into part of class?
def generate_score_metric_from_base(base_metric: BaseMetric):
    metrics = ScoreMetric.from_instance(base_metric)  # Inherit from the base metric

    epsilon = 0.000001
    
    def nonzero(maybe_zero):
        return max(maybe_zero, epsilon)

    tp = base_metric.tp
    tn = base_metric.tn
    fp = base_metric.fp
    fn = base_metric.fn

    metrics.accuracy = (tp + tn) / (tp + tn + fp + fn)
    metrics.precision  = tp / nonzero(tp + fp)  # Also called Positive Predictive Value (PPV)
    metrics.recall = tp / nonzero(tp + fn)  # Also called True positive rate or sensitivity
    metrics.f1 = 2 * (metrics.precision * metrics.recall) / nonzero(metrics.precision + metrics.recall)
    metrics.FRR = fn / nonzero(tp + fn)  # Also called FNR False negative rate
    metrics.FAR = fp / nonzero(tn + fp)  # Also called FPR False positive rate POSSIBLY WRONG, FAR might be fp / total
    # Specificity (SPC) or True Negative Rate (TNR):    # SPC = TN / N = TN / (FP + TN)
    # Negative Predictive Value (NPV):     # NPV = TN / (TN + FN)
    # Fall-Out or False Positive Rate (FPR):    # FPR = FP / N = FP / (FP + TN) =1 – SPC
    # False Discovery Rate (FDR):    # FDR = FP / (FP + TP) =1 – PPV  # Good pga ikke TN
    # Mathews Correlation Coefficient (MCC):    # MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP) *(TN+FN))

    return metrics

#
# Metrics
#


def compare_two_unique_desc_lists(margin: float, desc_list1, desc_list2):
    # for i in range(min(len(desc_list1), len(desc_list2))):  # Assert torch.all, torch.eq isnt working, doing it manually
    #     assert not torch.all(torch.eq(desc_list1[i], desc_list2[i]))
    within_margin = 0
    outside_margin = 0
    for desc1 in desc_list1:
        for desc2 in desc_list2:
            d = distance(desc1, desc2)
            if d < margin:
                within_margin += 1
            else:
                outside_margin += 1
    return within_margin, outside_margin


# Assumes dist(desc1, desc2) == dist(desc2, desc1)
def get_base_metric_all_vs_all(margin: float, descriptor_dict):
    metrics = BaseMetric(tp=0, fp=0, tn=0, fn=0)

    descriptors = []
    label_ident = []
    label_name = []

    # Collapse the dict to one list
    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            tmp2 = name.split("_")[1]
            if "03" in tmp2 or "04" in tmp2:
                continue
            descriptors.append(data)
            label_ident.append(ident)
            label_name.append(name)

    assert len(descriptors) == len(label_ident) == len(label_name)
    descriptors = torch.stack(descriptors)
    distances = onlineTripletLoss.pairwise_distances(embeddings=descriptors)

    margins = [0.1, 0.2, 0.5, 1.0, 1.5, 2, 3]

    for margin in margins:
        metrics = BaseMetric(tp=0, fp=0, tn=0, fn=0)
        # TODO make in matrix notation?
        # Check all vs all, same ID
        length = descriptors.size()[0]
        for idx1 in range(length):
            for idx2 in range(idx1+1, length):  # Do not check it against previous checked (Half of matrix used only)
                assert idx1 != idx2  # Do not check against itself
                assert label_name[idx1] != label_name[idx2]  # Dont use label name for other than verification


                d = distances[idx1, idx2]
                if label_ident[idx1] == label_ident[idx2]:
                    # Same Label, different ident
                    if d < margin:  # Good
                        metrics.tp += 1
                    else:           # Bad
                        metrics.fn += 1
                else:
                    # Different label
                    if d >= margin:  # Good
                        metrics.tn += 1
                    else:            # Bad
                        metrics.fp += 1
        print(f"margin:{margin}, {generate_score_metric_from_base(metrics)}")

    return metrics


# torch_geometric.utils.  accuracy, precision, recall, f1
@torch.no_grad()
def get_metric_all_vs_all(margin: float, descriptor_dict):
    base_metric = get_base_metric_all_vs_all(margin=margin, descriptor_dict=descriptor_dict)
    del descriptor_dict
    return generate_score_metric_from_base(base_metric)


def split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict):
    gallery_dict = {}
    probe_dict = {}
    for ident in descriptor_dict.keys():
        gallery_dict[ident] = {}
        probe_dict[ident] = {}

    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            expression_scale = name.split("_")[1]
            # If they are neutral or as neutral almost neutral, add them to gallery
            if "00" in expression_scale:
                gallery_dict[ident][name] = data
            else:
                probe_dict[ident][name] = data
    return gallery_dict, probe_dict

# Function to split the bu3dfe data
@torch.no_grad()
def get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict):
    gallery_dict, probe_dict = split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
    return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict)

def split_gallery_set_vs_probe_set_bosphorus(descriptor_dict):
    gallery_dict = {}
    probe_dict = {}
    for ident in descriptor_dict.keys():
        gallery_dict[ident] = {}
        probe_dict[ident] = {}

    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            # Two cases following the "Deep 3D Face identification"
            # 1. Neutral: _N_N_
            # 2. Expression: _E_
            if "_N_N_" in name:
                gallery_dict[ident][name] = data
            #elif "_E_" in name:
            #    probe_dict[ident][name] = data
            #else:
            #    raise RuntimeError("Invalid data?")  # Mostly for debug
            else: 
                probe_dict[ident][name] = data
    return gallery_dict, probe_dict

# Function to split the bosphorus data
@torch.no_grad()
def get_metric_gallery_set_vs_probe_set_bosphorus(descriptor_dict):
    gallery_dict, probe_dict = split_gallery_set_vs_probe_set_bosphorus(descriptor_dict)
    return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict, bosp=True)


def split_gallery_set_vs_probe_set_frgc(descriptor_dict):
    gallery_dict = {}
    probe_dict = {}
    for ident in descriptor_dict.keys():
        gallery_dict[ident] = {}
        probe_dict[ident] = {}

    for ident, ident_dict in descriptor_dict.items():
        ident_dict_items_sorted = sorted(ident_dict.items(), key=lambda item: int(item[0].split("d")[1]))  # item is of (name, data), sort by the "d" factor
        if len(ident_dict_items_sorted) > 1:
            # print("1", int(ident_dict_items_sorted[0][0].split("d")[1]), ident_dict_items_sorted[0][0])
            # print("2", int(ident_dict_items_sorted[1][0].split("d")[1]), ident_dict_items_sorted[1][0])
            if not int(ident_dict_items_sorted[0][0].split("d")[1]) < int(ident_dict_items_sorted[1][0].split("d")[1]):
                print(ident_dict_items_sorted[0][0], ident_dict_items_sorted[1][0])
                print(ident)
            assert int(ident_dict_items_sorted[0][0].split("d")[1]) < int(ident_dict_items_sorted[1][0].split("d")[1])

        # First ident in gallery
        first_ident = ident_dict_items_sorted.pop(0)
        gallery_dict[ident][first_ident[0]] = first_ident[1]

        # Rest in probe
        for name, data in ident_dict_items_sorted:
            probe_dict[ident][name] = data
    return gallery_dict, probe_dict

# Function to split the bosphorus data
@torch.no_grad()
def get_metric_gallery_set_vs_probe_set_frgc(descriptor_dict):
    gallery_dict, probe_dict = split_gallery_set_vs_probe_set_frgc(descriptor_dict)
    return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict)


def get_metric_gallery_set_vs_probe_set(gallery_descriptors_dict, probe_descriptors_dict, bosp=False):
    
    gal_descriptors = []
    gal_ident = []
    gal_name = []
    for ident, ident_dict in gallery_descriptors_dict.items():
            for name, data in ident_dict.items():
                gal_descriptors.append(data)
                gal_ident.append(ident)
                gal_name.append(name)

    probe_descriptors = []
    probe_ident = []
    probe_name = []
    for ident, ident_dict in probe_descriptors_dict.items():
            for name, data in ident_dict.items():
                probe_descriptors.append(data)
                probe_ident.append(ident)
                probe_name.append(name)


    gal_descriptors = torch.squeeze(torch.stack(gal_descriptors))
    probe_descriptors = torch.squeeze(torch.stack(probe_descriptors))

    # Want to create a matrix, where each row is for a garllery descriptor, and each col is for a probe descriptor 
    distances = torch.cdist(gal_descriptors, probe_descriptors, p=2)

    # Transpose to make the math prettier, so now its [probe, gal]  TODO switch the generation instead?
    distances = torch.transpose(distances, 0, 1)
    # for each probe ident, find the rank-1
    idxs_min = torch.argmin(distances, dim=1)  # Iterate over all rows, and find the indices of the smallest number

    metrics = BaseMetric(tp=0, fp=0, tn=0, fn=0)

    exprs = ["_N_N_", "_LFAU_", "_UFAU_", "_CAU_", "_E_", "_YR_", "_PR_", "_CR_", "_O_"]
    def getxpr(name):
        for expr in exprs:
            if expr in name: return expr
        return "FUCKED UP" + name  # This willl crash, as it it not a valid key
    total_per_exprs = {}
    tp_per_exprs = {}
    for expr in exprs:
        total_per_exprs[expr] = 0
        tp_per_exprs[expr] = 0

    for probe_idx, min_gal_idx in enumerate(idxs_min):
        assert probe_name[probe_idx] != gal_name[min_gal_idx]  # Assert that the same face is not in both datasets
        if probe_ident[probe_idx] == gal_ident[min_gal_idx]:
            metrics.tp += 1  # Good, correct ident
            if bosp:
                tp_per_exprs[getxpr(probe_name[probe_idx])] += 1
        else:
            metrics.fp += 1  # Bad, not corrent ident
        if bosp:
            total_per_exprs[getxpr(probe_name[probe_idx])] += 1
    
    if bosp:
        buildup = ""
        for exp in exprs:
            buildup += f"{exp[1:-1]}: {tp_per_exprs[exp]}/{total_per_exprs[exp]} ({tp_per_exprs[exp]/(total_per_exprs[exp]+1):.3f}), "
        print(buildup[:-2])
    # TODO fix abusement of base metric
    return generate_score_metric_from_base(metrics)

def generate_flat_descriptor_dict(descriptor_dict, device):
    # Generate flatter dict, with labels and data
    datas = []
    labels = []
    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            datas.append(data)
            # Encode name, in case it is not an int
            #   Assumes 0-9a-zA-Z, "F0054" becomes 32'10'10'15'14
            ints = [str(ord(c)-38) for c in str(ident)]  # Between 10 (for 0) and 84(z)
            int_ident = int("".join(ints))
            labels.append(int_ident)  # assumes number
    datas = torch.stack(datas).to(device)
    labels = torch.tensor(labels).to(device)
    return datas, labels


@torch.no_grad()
def get_siamese_rank1_gallery_set_vs_probe_set_BU3DFE(siamese, device, criterion, descriptor_dict):
    gallery_dict, probe_dict = split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
    return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)

@torch.no_grad()
def get_siamese_rank1_gallery_set_vs_probe_set_bosphorus(siamese, device, criterion, descriptor_dict):
    gallery_dict, probe_dict = split_gallery_set_vs_probe_set_bosphorus(descriptor_dict)
    return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)
    
@torch.no_grad()
def get_siamese_rank1_gallery_set_vs_probe_set_frgc(siamese, device, criterion, descriptor_dict):
    gallery_dict, probe_dict = split_gallery_set_vs_probe_set_frgc(descriptor_dict)
    return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)

@torch.no_grad()
def generate_metirc_siamese_rank1(siamese, device, criterion, gallery_descriptors_dict, probe_descriptors_dict):
    siamese.eval()

    gal_datas, gal_labels = generate_flat_descriptor_dict(gallery_descriptors_dict, device)
    probe_datas, probe_labels = generate_flat_descriptor_dict(probe_descriptors_dict, device)
    # print(gal_datas.shape)  # ([20, 128]) or 80
    # print(gal_labels.shape)  # 20 or 80
    # print(probe_datas.shape)  # torch.Size([480, 128]) or 1920
    # print(probe_labels.shape)  # 480 or 1820

    # Input1 must be [gal, probe, gal-desc]
    # Input2 must be [gal, probe, probe-desc]
    # Want output to be matrix of [gal, probe, 1] to use max on

    # Add singleton dimension before expansiopn (instead of transpose)
    datas1 = gal_datas.unsqueeze(1).expand(-1, probe_datas.shape[0], -1) # gal, probe gal-desc
    datas2 = probe_datas.expand(gal_datas.shape[0], -1, -1) # gal, probe, probe-desc
    # print("datas1", datas1.shape)  # [20, 480, 128], or [80, 1920, 128]
    # print("datas2", datas2.shape)  # [20, 480, 128], or [80, 1920, 128]

    # Labels of type [gal, probe] -> id
    gal_labels_expanded = gal_labels.unsqueeze(1).expand(-1, probe_labels.shape[0])
    probe_labels_exapnded = probe_labels.expand(gal_labels.shape[0], -1)
    # [gal, probe] -> true/false if its a valid pair
    correct_identity_matrix = gal_labels_expanded.eq(probe_labels_exapnded)
    # print("ident", correct_identity_matrix.shape)  # [20, 480] or [80, 1920]

    result_matrix = siamese(datas1, datas2)
    loss = criterion(result_matrix, correct_identity_matrix.float())

    # Make it [probe, gal]
    result_matrix = torch.transpose(result_matrix, 0, 1)
    # Find best ident
    idxs_max = torch.argmax(result_matrix, dim=1)  # "Check each row"

    metrics = BaseMetric(tp=0, fp=0, tn=0, fn=0)

    # TODO make into matrix sytax
    for probe_idx, max_gal_idx in enumerate(idxs_max):
        if correct_identity_matrix[max_gal_idx, probe_idx]:  # correct id matrix is still gal, probe
            metrics.tp += 1
        else: 
            metrics.fp += 1

    return loss.item(), generate_score_metric_from_base(metrics)

@torch.no_grad()
def generate_metric_siamese(siamese, device, criterion, descriptor_dict):
    siamese.eval()
    # Want to match each id to each other id

    # Generate flatter dict, with labels and data
    datas, labels = generate_flat_descriptor_dict(descriptor_dict, device)

    # This also creates pairs of the same descriptor. Beware if it messes with metric
    labels_combi = torch.combinations(labels, r=2, with_replacement=True)
    labels_1d = labels_combi[:, 0].eq(labels_combi[:, 1])

    indecies = torch.arange(datas.shape[0]).to(device)
    indecies = torch.combinations(indecies, r=2, with_replacement=True)
    assert indecies.shape[0] == labels_1d.shape[0]

    # Find indencies for all positive and negative pairs
    indecies_pos = labels_1d.eq(1)
    indecies_neg = labels_1d.eq(0)

    descriptors_combi_1 = datas[indecies[:, 0]]
    descriptors_combi_2 = datas[indecies[:, 1]]

    max_size = 150000
    splitted_desc1 = torch.split(descriptors_combi_1, max_size)
    splitted_desc2 = torch.split(descriptors_combi_2, max_size)

    results = []
    for i in range(len(splitted_desc1)):
        short_result = siamese(splitted_desc1[i], splitted_desc2[i])
        results.append(short_result)

    result = torch.cat(results, dim=0)
    loss = criterion(result, labels_1d.float())

    correct = (result>0.5).eq(labels_1d).sum().item()
    total = result.shape[0]
    tp = (result[indecies_pos]>0.5).eq(labels_1d[indecies_pos]).sum().item()
    tn = (result[indecies_neg]>0.5).eq(labels_1d[indecies_neg]).sum().item()
    fn = result[indecies_pos].shape[0] - tp
    fp = result[indecies_neg].shape[0] - tn
    assert (tp+tn+fn+fp) == total

    metric = BaseMetric(tp, fp, tn, fn)
    return loss.item(), generate_score_metric_from_base(metric)

@torch.no_grad()
def generate_descriptor_dict_from_dataloader(model, dataloader, device):
    model.eval()

    descriptor_dict = {}
    for larger_batch in dataloader:
        for batch in larger_batch:
            batch = batch.to(device)
            output = model(batch).to("cpu")

            ids = batch.dataset_id
            names = batch.name
            for idx, id in enumerate(ids):
                if id not in descriptor_dict:
                    descriptor_dict[id] = {}
                descriptor_dict[id][names[idx]] = output[idx]

    return descriptor_dict


if __name__ == "__main__":
    gallery_descriptors = torch.tensor([[1.0, 1.0], [2.0,2.0], [4.0,4.0], [5.0, 5.0]])
    print("gal", gallery_descriptors.shape)
    probe_descriptors = torch.tensor([[0.0, 0.0], [3.0,2.0]])
    print("prob", probe_descriptors.shape)

    distances = torch.cdist(gallery_descriptors, probe_descriptors, p=2)
    print(distances)

    for i in range(len(gallery_descriptors)):
        for j in range(len(probe_descriptors)):
            value = torch.isclose(distances[i][j], torch.dist(gallery_descriptors[i], probe_descriptors[j])).item()
            assert(value)

    distances = torch.transpose(distances, 0, 1)
    print(distances)
    idxs_min = torch.argmin(distances, dim=1)
    print(idxs_min)
