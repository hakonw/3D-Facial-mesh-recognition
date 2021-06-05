from numpy.core.numeric import indices
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
        has_one_neutral = False
        for name, data in ident_dict.items():
            expression_scale = name.split("_")[1]
            # If they are neutral or as neutral almost neutral, add them to gallery
            if "00" in expression_scale:
                gallery_dict[ident][name] = data
                assert not has_one_neutral
                has_one_neutral = True
            else:
                probe_dict[ident][name] = data
    return gallery_dict, probe_dict

# Function to split the bu3dfe data
# @torch.no_grad()
# def get_metric_gallery_set_vs_probe_set_BU3DFE(descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
#     return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict)

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
            if "_N_N_0" in name:  # Only allow the first neural image
                gallery_dict[ident][name] = data
            #elif "_E_" in name:
            #    probe_dict[ident][name] = data
            #else:
            #    raise RuntimeError("Invalid data?")  # Mostly for debug
            else: 
                probe_dict[ident][name] = data
        assert len(gallery_dict[ident].keys()) == 1, f"failed for {ident}. Dict: {gallery_dict[ident].keys()}, input: {ident_dict.keys()}"
        assert len(probe_dict[ident].keys()) > 1
    return gallery_dict, probe_dict

# # Function to split the bosphorus data
# @torch.no_grad()
# def get_metric_gallery_set_vs_probe_set_bosphorus(descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_bosphorus(descriptor_dict)
#     return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict, bosp=True)


def split_gallery_set_vs_probe_set_frgc(descriptor_dict):
    gallery_dict = {}
    probe_dict = {}
    for ident in descriptor_dict.keys():
        gallery_dict[ident] = {}
        probe_dict[ident] = {}

    for ident, ident_dict in descriptor_dict.items():
        ident_dict_items_sorted = sorted(ident_dict.items(), key=lambda item: int(item[0].split("d")[1]))  # item is of (name, data), sort by the "d" factor
        assert int(ident_dict_items_sorted[0][0].split("d")[1]) == min([int(it[0].split("d")[1]) for it in ident_dict_items_sorted])
        if len(ident_dict_items_sorted) > 1:
            assert int(ident_dict_items_sorted[0][0].split("d")[1]) < int(ident_dict_items_sorted[1][0].split("d")[1])

        # First ident in gallery
        first_ident = ident_dict_items_sorted.pop(0)
        gallery_dict[ident][first_ident[0]] = first_ident[1]

        # Rest in probe
        for name, data in ident_dict_items_sorted:
            probe_dict[ident][name] = data
    return gallery_dict, probe_dict

# Function to split the bosphorus data
# @torch.no_grad()
# def get_metric_gallery_set_vs_probe_set_frgc(descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_frgc(descriptor_dict)
#     return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict)


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

def split_gallery_set_vs_probe_set_3dface(descriptor_dict):
    gallery_dict = {}
    probe_dict = {}
    for ident in descriptor_dict.keys():  # Prebuild
        gallery_dict[ident] = {}
        probe_dict[ident] = {}

    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            # Two cases 
            # 1. _0
            # 2. _1
            assert "_0" in name or "_1" in name
            if "_0" in name:
                gallery_dict[ident][name] = data
            else: 
                probe_dict[ident][name] = data
    return gallery_dict, probe_dict

# @torch.no_grad()
# def get_metric_gallery_set_vs_probe_set_3dface(descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_3dface(descriptor_dict)
#     return get_metric_gallery_set_vs_probe_set(gallery_dict, probe_dict)

# @torch.no_grad()
# def get_siamese_rank1_gallery_set_vs_probe_set_3dface(siamese, device, criterion, descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_3dface(descriptor_dict)
#     return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)

# @torch.no_grad()
# def get_siamese_rank1_gallery_set_vs_probe_set_BU3DFE(siamese, device, criterion, descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_BU3DFE(descriptor_dict)
#     return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)

# @torch.no_grad()
# def get_siamese_rank1_gallery_set_vs_probe_set_bosphorus(siamese, device, criterion, descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_bosphorus(descriptor_dict)
#     return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)
    
# @torch.no_grad()
# def get_siamese_rank1_gallery_set_vs_probe_set_frgc(siamese, device, criterion, descriptor_dict):
#     gallery_dict, probe_dict = split_gallery_set_vs_probe_set_frgc(descriptor_dict)
#     return generate_metirc_siamese_rank1(siamese, device, criterion, gallery_dict, probe_dict)

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

    for probe_idx, max_gal_idx in enumerate(idxs_max):
        if correct_identity_matrix[max_gal_idx, probe_idx]:  # correct id matrix is still gal, probe
            metrics.tp += 1
        else: 
            metrics.fp += 1

    # if rank > 1:
    #     idx_max_topk = torch.topk(result_matrix, rank, dim=1)[1]  # returns (values, indices)
    #     for probe_idx, max_gal_idxs in enumerate(idx_max_topk):  # get each row index, and for that, each argmax-list for that row
    #         found_valid = False
    #         for max_gal_idx in max_gal_idxs:
    #             if correct_identity_matrix[max_gal_idx, probe_idx]:  # correct id matrix is still gal, probe
    #                 metrics.tp += 1
    #                 found_valid = True
    #                 break
    #         if not found_valid:
    #             metrics.fp += 1

    return loss.item(), generate_score_metric_from_base(metrics)

@torch.no_grad()
def generate_metric_siamese(siamese, device, criterion, descriptor_dict, offload=False):
    siamese.eval()
    # Want to match each id to each other id

    # Generate flatter dict, with labels and data
    datas, labels = generate_flat_descriptor_dict(descriptor_dict, device)
    if offload:
        datas = datas.cpu()

    # This also creates pairs of the same descriptor. Beware if it messes with metric
    labels_combi = torch.combinations(labels, r=2, with_replacement=True)
    labels_1d = labels_combi[:, 0].eq(labels_combi[:, 1])

    # Generate the pairs of all indicies (used by descriptors)
    indecies = torch.arange(datas.shape[0]).to(device)
    indecies = torch.combinations(indecies, r=2, with_replacement=True)
    assert indecies.shape[0] == labels_1d.shape[0]

    # Find indencies for all positive and negative pairs
    indecies_pos = labels_1d.eq(1)
    indecies_neg = labels_1d.eq(0)

    max_size = 100000
    splitted_desc1 = torch.split(datas[indecies[:, 0]], max_size)
    splitted_desc2 = torch.split(datas[indecies[:, 1]], max_size)


    results = []
    for i in range(len(splitted_desc1)):
        short_result = siamese(splitted_desc1[i].to(device), splitted_desc2[i].to(device))
        results.append(short_result)

    pred = torch.cat(results, dim=0)
    loss = criterion(pred, labels_1d.float())

    correct = (pred>0.5).eq(labels_1d).sum().item()
    total = pred.shape[0]
    tp = (pred[indecies_pos]>0.5).eq(labels_1d[indecies_pos]).sum().item()
    tn = (pred[indecies_neg]>0.5).eq(labels_1d[indecies_neg]).sum().item()
    fn = pred[indecies_pos].shape[0] - tp
    fp = pred[indecies_neg].shape[0] - tn
    assert (tp+tn+fn+fp) == total

    metric = BaseMetric(tp, fp, tn, fn)
    return loss.item(), generate_score_metric_from_base(metric), pred, labels_1d.float()

import sklearn.metrics

@torch.no_grad()
def generate_metirc_siamese_rank1_cmc(siamese, device, gallery_descriptors_dict, probe_descriptors_dict):
    siamese.eval()

    gal_datas, gal_labels = generate_flat_descriptor_dict(gallery_descriptors_dict, device)
    probe_datas, probe_labels = generate_flat_descriptor_dict(probe_descriptors_dict, device)

    # Add singleton dimension before expansiopn (instead of transpose)
    datas1 = gal_datas.unsqueeze(1).expand(-1, probe_datas.shape[0], -1) # gal, probe, gal-desc
    datas2 = probe_datas.expand(gal_datas.shape[0], -1, -1)              # gal, probe, probe-desc

    # Labels of type [gal, probe] -> id
    gal_labels_expanded = gal_labels.unsqueeze(1).expand(-1, probe_labels.shape[0])
    probe_labels_exapnded = probe_labels.expand(gal_labels.shape[0], -1)
    # [gal, probe] -> true/false if its a valid pair
    correct_identity_matrix = gal_labels_expanded.eq(probe_labels_exapnded)

    result_matrix = siamese(datas1, datas2)
    # loss = criterion(result_matrix, correct_identity_matrix.float())

    # Make it [probe, gal]
    result_matrix = torch.transpose(result_matrix, 0, 1)
    
    rank_tp_dict = {}
    def add(rank):
        if rank not in rank_tp_dict:
            rank_tp_dict[rank] = 1
        else:
            rank_tp_dict[rank] += 1

    # Find best idents
    sorted_results, sorted_results_idx = torch.sort(result_matrix, dim=1, descending=True)  # Get the highest values first

    # TODO this can most likely be done more efficently with matrixes
    for probe_idx in range(sorted_results_idx.shape[0]):
        org_gal_idxs = sorted_results_idx[probe_idx]  # Each original indicie, where the first one is the highest value
        for r, gal_idx in enumerate(org_gal_idxs):
            if correct_identity_matrix[gal_idx, probe_idx]:
                add(r)
                break
        else:
            print("what")
            # assert False, "Did not find the probe in the gallery, ergo not a closed set"
    
    return rank_tp_dict

@torch.no_grad()
def generate_metric_siamese_roc(siamese, device, gallery_descriptors_dict, probe_descriptors_dict):
    print("Did you mean to use balanced?")
    siamese.eval()

    gal_datas, gal_labels = generate_flat_descriptor_dict(gallery_descriptors_dict, device)
    probe_datas, probe_labels = generate_flat_descriptor_dict(probe_descriptors_dict, device)

    # Add singleton dimension before expansiopn (instead of transpose)
    datas1 = gal_datas.unsqueeze(1).expand(-1, probe_datas.shape[0], -1) # gal, probe, gal-desc
    datas2 = probe_datas.expand(gal_datas.shape[0], -1, -1)              # gal, probe, probe-desc

    # Labels of type [gal, probe] -> id
    gal_labels_expanded = gal_labels.unsqueeze(1).expand(-1, probe_labels.shape[0])  # gal, probe
    probe_labels_exapnded = probe_labels.expand(gal_labels.shape[0], -1)             # gal, probe
    # [gal, probe] -> true/false if its a valid pair
    correct_identity_matrix = gal_labels_expanded.eq(probe_labels_exapnded)

    # [gal, probe]
    result_matrix = siamese(datas1, datas2)

    # Make it [probe, gal]
    # result_matrix = torch.transpose(result_matrix, 0, 1)
    
    # Flatten for use in roc
    y_true = correct_identity_matrix.flatten()
    y_score = result_matrix.flatten()

    assert y_true.shape[0] == y_score.shape[0]

    return y_true, y_score

@torch.no_grad()
def generate_metric_siamese_roc_bal(siamese, device, gallery_descriptors_dict, probe_descriptors_dict):
    siamese.eval()

    gal_datas, gal_labels = generate_flat_descriptor_dict(gallery_descriptors_dict, device)
    probe_datas, probe_labels = generate_flat_descriptor_dict(probe_descriptors_dict, device)

    # Add singleton dimension before expansiopn (instead of transpose)
    datas1 = gal_datas.unsqueeze(1).repeat(1, probe_datas.shape[0], 1) # size: gal, probe, gal-desc
    datas2 = probe_datas.repeat(gal_datas.shape[0], 1, 1)              # size: gal, probe, probe-desc

    # Labels of type [gal, probe] -> id
    gal_labels_expanded = gal_labels.unsqueeze(1).repeat(1, probe_labels.shape[0])  # size: gal, probe
    probe_labels_exapnded = probe_labels.repeat(gal_labels.shape[0], 1)             # size: gal, probe
    # [gal, probe] -> true/false if its a valid pair
    correct_identity_matrix = gal_labels_expanded.eq(probe_labels_exapnded)

    # Balancing

    # Create a flat tensor of all correct labels
    flat_correct_labels = correct_identity_matrix[correct_identity_matrix.eq(1)]  
    assert flat_correct_labels.shape[0] == correct_identity_matrix.sum()
    # print("amount of 1 in correct_ident_matrix", correct_identity_matrix.sum()) # 1920
    # print("flat correct ident shape", flat_correct_labels.shape) # size: [1920]

    # filter out all positive pairs
    only_pos_pairs_datas1 = datas1[correct_identity_matrix, :]
    only_pos_pairs_datas2 = datas2[correct_identity_matrix, :]
    assert only_pos_pairs_datas1.shape[0] == flat_correct_labels.shape[0]
    assert only_pos_pairs_datas2.shape[0] == flat_correct_labels.shape[0]
    # print("onlyposdatas1", only_pos_pairs_datas1.shape) # torch.Size([1920, 128])
    # print("onlyposdatas2", only_pos_pairs_datas2.shape) # torch.Size([1920, 128])

    # filter out all negative pairs
    only_neg_pairs_datas1 = datas1[correct_identity_matrix.eq(0), :]
    only_neg_pairs_datas2 = datas2[correct_identity_matrix.eq(0), :]
    assert only_neg_pairs_datas1.shape[0] == correct_identity_matrix.eq(0).sum()
    assert only_neg_pairs_datas2.shape[0] == correct_identity_matrix.eq(0).sum()
    # print("onlynegdatas1", only_neg_pairs_datas1.shape) # torch.Size([151680, 128])
    # print("onlynegdatas2", only_neg_pairs_datas2.shape) # torch.Size([151680, 128])

    # Select the same amount of negatiive pairs as positive
    # Also save the RNG for reproducibility
    amount_positive_pairs = flat_correct_labels.shape[0]
    t_seed = torch.get_rng_state()
    t_seed_gpu = torch.cuda.get_rng_state(device)
    t_seed_rand = torch.random.get_rng_state()
    torch.random.manual_seed(1); torch.cuda.manual_seed(1); torch.manual_seed(1)
    perm = torch.randperm(only_neg_pairs_datas1.shape[0], device=device)
    torch.random.set_rng_state(t_seed_rand)
    torch.cuda.set_rng_state(t_seed_gpu, device)
    torch.set_rng_state(t_seed)

    idx = perm[:amount_positive_pairs]
    only_neg_pairs_datas1_sampled = only_neg_pairs_datas1[idx, :]
    only_neg_pairs_datas2_sampled = only_neg_pairs_datas2[idx, :]
    assert only_neg_pairs_datas1_sampled.shape[0] == amount_positive_pairs
    assert only_neg_pairs_datas2_sampled.shape[0] == amount_positive_pairs
    # print("neg sampled1", only_neg_pairs_datas1_sampled.shape)
    # print("neg sampled2", only_neg_pairs_datas2_sampled.shape)

    combined_datas_1 = torch.cat([only_pos_pairs_datas1, only_neg_pairs_datas1_sampled], 0)
    combined_datas_2 = torch.cat([only_pos_pairs_datas2, only_neg_pairs_datas2_sampled], 0)
    assert combined_datas_1.shape[0] == amount_positive_pairs*2
    assert combined_datas_2.shape[0] == amount_positive_pairs*2
    # print("combined1", combined_datas_1.shape) # torch.Size([3840, 128])
    # print("combined2", combined_datas_2.shape) # torch.Size([3840, 128])

    # Safely creating the labels
    flat_incorrect_labels = torch.zeros([amount_positive_pairs], device=device)
    combined_labels = torch.cat([flat_correct_labels, flat_incorrect_labels], 0)
    assert combined_labels[:combined_labels.shape[0]//2].sum() == amount_positive_pairs
    assert combined_labels[combined_labels.shape[0]//2:].sum() == 0


    # [N positive pairs + N negative pairs, descriptors]
    result_matrix = siamese(combined_datas_1, combined_datas_2)

    # Flatten
    y_score = result_matrix.flatten()
    y_true = combined_labels

    assert y_true.shape[0] == y_score.shape[0]

    return y_true, y_score

@torch.no_grad()
def generate_metric_siamese_roc_bal_all(siamese, device, dict):
    siamese.eval()

    datas, labels = generate_flat_descriptor_dict(dict, device)

        
    indecies = torch.arange(datas.shape[0]).to(device); # Create indicies for all possible conbinations
    indecies = torch.combinations(indecies, r=2, with_replacement=True)
    labels_combi = labels[indecies[:, 0]].eq(labels[indecies[:, 1]])

    # balance
    # Find indecies for all positive.
    indecies_pos = indecies[labels_combi.eq(1)]
    label_pos = labels_combi[labels_combi.eq(1)]

    # Find indecies for alle negative pairs
    t_seed = torch.get_rng_state(); t_seed_gpu = torch.cuda.get_rng_state(device); t_seed_rand = torch.random.get_rng_state()
    torch.random.manual_seed(1); torch.cuda.manual_seed(1); torch.manual_seed(1)
    mask_neg = torch.randperm(indecies[labels_combi.eq(0)].shape[0])[:indecies_pos.shape[0]]  # Create a random mask of the same size as indicies pos
    torch.random.set_rng_state(t_seed_rand); torch.cuda.set_rng_state(t_seed_gpu, device); torch.set_rng_state(t_seed)

    indecies_neg = indecies[labels_combi.eq(0)][mask_neg]
    label_neg = labels_combi[labels_combi.eq(0)][mask_neg]
    assert indecies_pos.shape[0] == indecies_neg.shape[0]
    assert indecies_pos.shape[0] == label_pos.shape[0]
    assert label_pos.shape[0] == label_neg.shape[0]

     # Cobmine for all indecies to give to the model
    indecies = torch.cat((indecies_pos, indecies_neg), dim=0)
    labels_combi = torch.cat((label_pos, label_neg), dim=0)

    middle = indecies_pos.shape[0]  # middle = labels_combi.shape[0]//2 becaome wrong when it was odd
    assert labels_combi[:middle].sum().eq(middle).item()  # The first half should all be "1"
    assert labels_combi[middle:].sum().eq(0).item()  # The seconds half should all be "0"

    descriptors_combi_1 = datas[indecies[:, 0]]
    descriptors_combi_2 = datas[indecies[:, 1]]

    # [N positive pairs + N negative pairs, descriptors]
    result = siamese(descriptors_combi_1, descriptors_combi_2)

    # Flatten
    y_score = result.flatten()
    y_true = labels_combi

    assert y_true.shape[0] == y_score.shape[0]
    return y_true, y_score

@torch.no_grad()
def filter_descriptordict_scans(descriptor_dict, allow_filter):
    new_dict = {}
    for ident, ident_dict in descriptor_dict.items():
        for name, data in ident_dict.items():
            if allow_filter(name):
                if ident not in new_dict:
                    new_dict[ident] = {}
                new_dict[ident][name] = data
    return new_dict

@torch.no_grad()
def generate_identification_rate_by_rank(rank_tp_dict):
    total = sum(rank_tp_dict.values())  # Used to calculate Identification rate
    # sorted_dict = sorted(rank_tp_dict.items(), key=lambda x: x[0])
    max_rank = max(rank_tp_dict.keys())

    cummulative = 0
    identification_rate_by_rank = []
    for i in range(max_rank + 1):  # for i from 0 to highest rank value
        if i in rank_tp_dict:
            cummulative += rank_tp_dict[i]
        identification_rate_by_rank.append(cummulative/total)  # Calculate identification rate for the cummulative value
    # print(identification_rate_by_rank)
    return identification_rate_by_rank

@torch.no_grad()
def generate_cmc(siamese, device, gallery_descriptors_dict, probe_descriptors_dict):
    rank_tp_dict = generate_metirc_siamese_rank1_cmc(siamese, device, gallery_descriptors_dict, probe_descriptors_dict)
    identification_rate_by_rank = generate_identification_rate_by_rank(rank_tp_dict)
    
    fig = plt.figure()
    plt.title("CMC")
    plt.plot(list(range(1, len(identification_rate_by_rank)+1)), identification_rate_by_rank, color="darkorange", marker="o", clip_on=False, label = "TODO")  # b
    plt.ylabel("Identification Rate")
    plt.xlabel("Rank")
    plt.legend(loc="lower right")
    plt.xlim(left=1, right=len(identification_rate_by_rank))
    plt.locator_params(axis="x", integer=True)
    plt.grid(True)
    return fig

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import io
import PIL.Image 
from torchvision.transforms import ToTensor
@torch.no_grad()
def generate_roc(y_true, y_score):
    y_true = y_true.cpu()
    y_score = y_score.cpu()
    false_positive_rate, recall, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(false_positive_rate, recall)
    fig = plt.figure()  # fig, ax = plt.subplots
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.plot(false_positive_rate, recall, color="darkorange", label = "AUC = %0.3f" % roc_auc)  # b
    plt.legend(loc="lower right")
    plt.plot([0,1], [0,1], lw=2, linestyle="--")  # "r--"
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False positive Rate (1-Specificity) (FAR-false acceptance rate)")
    plt.grid(True)
    return fig
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image)
    # return image



@torch.no_grad()
def generate_descriptor_dict_from_dataloader(model, dataloader, device):
    model.eval()

    descriptor_dict = {}
    for larger_batch in dataloader:
        for batch in larger_batch:
            batch = batch.to(device)
            output = model(batch).to("cpu")

            ids = batch.dataset_id  # Real ID, not generated number id
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
